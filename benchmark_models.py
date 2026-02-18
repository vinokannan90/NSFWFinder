"""
Model Benchmark
===============
Compare NSFW detection accuracy across three models on the same set of images.

Models tested:
  1. Falconsai/nsfw_image_detection   — Current model (ViT-base, 224 px input)
  2. Marqo/nsfw-image-detection-384   — EfficientNet-based (384 px input)
  3. SigLIP zero-shot                 — Vision-language model with text prompts (384 px)

How to use:
  1. Create a folder with test images — ideally:
       • Some images that ARE genuinely NSFW  (true positives)
       • Some that are NOT NSFW but were wrongly flagged by Falconsai
         (false positives — close-ups, portraits, skin-toned objects, etc.)
  2. Run:
       python benchmark_models.py <folder>

  The script scores every image with all three models, prints a side-by-side
  comparison table, and saves results to a CSV for easy review.

Usage:
    python benchmark_models.py <image_folder> [options]

Examples:
    python benchmark_models.py C:\\TestImages
    python benchmark_models.py C:\\TestImages --threshold 0.8 --output my_benchmark.csv
    python benchmark_models.py C:\\TestImages --cpu
    python benchmark_models.py C:\\TestImages --models falconsai marqo
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Protocol, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp",
}

MODEL_CHOICES = {"falconsai", "marqo", "siglip"}


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

class NSFWModel(Protocol):
    NAME: str
    def score(self, image: Image.Image) -> float: ...


# ---------------------------------------------------------------------------
# Model 1 — Falconsai/nsfw_image_detection  (ViT-base, 224 px)
# ---------------------------------------------------------------------------

class FalconsaiModel:
    NAME = "Falconsai (ViT-base)"

    def __init__(self, device: torch.device):
        from transformers import AutoModelForImageClassification, ViTImageProcessor

        repo = "Falconsai/nsfw_image_detection"
        print(f"  Loading {repo} …")
        self.processor = ViTImageProcessor.from_pretrained(repo)
        self.model = AutoModelForImageClassification.from_pretrained(repo)
        self.model.eval()
        self.device = device

        self._use_half = device.type == "cuda"
        if self._use_half:
            self.model = self.model.half()
        self.model.to(device)

        self._nsfw_idx = int(self.model.config.label2id.get("nsfw", 1))

    @torch.inference_mode()
    def score(self, image: Image.Image) -> float:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self._use_half:
            inputs = {
                k: v.half() if v.is_floating_point() else v
                for k, v in inputs.items()
            }
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self._nsfw_idx].cpu().item()


# ---------------------------------------------------------------------------
# Model 2 — Marqo/nsfw-image-detection-384  (EfficientNet-based, 384 px)
# ---------------------------------------------------------------------------

class MarqoModel:
    NAME = "Marqo (EfficientNet)"

    def __init__(self, device: torch.device):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        repo = "hf-hub:Marqo/nsfw-image-detection-384"
        print(f"  Loading {repo} …")
        self.model = timm.create_model(repo, pretrained=True)
        self.model.eval()
        self.device = device

        self._use_half = device.type == "cuda"
        if self._use_half:
            self.model = self.model.half()
        self.model.to(device)

        # Build the pre-processing pipeline for this model
        data_cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = create_transform(**data_cfg, is_training=False)

        # Determine label ordering ------------------------------------------------
        # Marqo model outputs 2 classes.  We try to auto-detect which index is
        # "nsfw" from the model config; if that information isn't available we
        # default to index 0 (most common convention for this model).
        self._nsfw_idx = self._detect_nsfw_index()
        print(f"    Marqo NSFW class index: {self._nsfw_idx}")

    def _detect_nsfw_index(self) -> int:
        """Try to read label mapping from the timm model config."""
        try:
            cfg = self.model.pretrained_cfg
            label_names = cfg.get("label_names", None) or cfg.get("labels", None)
            if label_names:
                for i, name in enumerate(label_names):
                    if "nsfw" in str(name).lower():
                        return i
        except Exception:
            pass
        # Default: index 1 is nsfw for most binary classifiers
        return 1

    @torch.inference_mode()
    def score(self, image: Image.Image) -> float:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self._use_half:
            tensor = tensor.half()
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self._nsfw_idx].cpu().item()


# ---------------------------------------------------------------------------
# Model 3 — SigLIP zero-shot  (ViT-SO400M, 384 px)
# ---------------------------------------------------------------------------

class SigLIPModel:
    NAME = "SigLIP (zero-shot)"

    # Prompt sets — tuned for distinguishing explicit content from innocent
    # close-ups, portraits, and skin-toned objects.
    NSFW_PROMPTS = [
        "an explicit sexual photograph",
        "a pornographic image",
        "a nude person in a sexual pose",
        "sexually explicit adult content",
    ]
    SAFE_PROMPTS = [
        "a normal photograph",
        "a portrait photo of a person's face",
        "a landscape, object, or food photo",
        "a safe family-friendly image",
    ]

    def __init__(self, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        repo = "google/siglip-so400m-patch14-384"
        print(f"  Loading {repo} …  (this model is ~3.6 GB, first download may take a while)")
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo)
        self.model.eval()
        self.device = device

        self._use_half = device.type == "cuda"
        if self._use_half:
            self.model = self.model.half()
        self.model.to(device)

        self._all_prompts = self.NSFW_PROMPTS + self.SAFE_PROMPTS
        self._n_nsfw = len(self.NSFW_PROMPTS)

    @torch.inference_mode()
    def score(self, image: Image.Image) -> float:
        inputs = self.processor(
            text=self._all_prompts,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self._use_half:
            inputs = {
                k: v.half() if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        outputs = self.model(**inputs)
        logits = outputs.logits_per_image  # (1, num_prompts)
        scores = torch.sigmoid(logits[0])  # SigLIP uses sigmoid, not softmax

        nsfw_max = scores[: self._n_nsfw].max().cpu().item()
        safe_max = scores[self._n_nsfw :].max().cpu().item()

        # Normalise to a 0–1 score comparable with the other models
        return nsfw_max / (nsfw_max + safe_max + 1e-8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(folder: str) -> List[str]:
    """Collect image files from *folder* (non-recursive)."""
    images = []
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            full = os.path.join(folder, fname)
            if os.path.isfile(full):
                images.append(full)
    images.sort()
    return images


def load_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        print(f"  [WARN] Cannot load: {path} ({exc})")
        return None


def flag_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "NSFW"
    elif score >= threshold * 0.8:
        return "borderline"
    else:
        return "safe"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    image_folder: str,
    model_keys: List[str],
    threshold: float,
    device: torch.device,
    output_csv: str,
) -> None:
    # ---- Collect images -----------------------------------------------------
    image_paths = collect_images(image_folder)
    if not image_paths:
        print(f"No images found in '{image_folder}'.")
        sys.exit(1)
    print(f"\nFound {len(image_paths)} images in '{image_folder}'.\n")

    # ---- Load models --------------------------------------------------------
    print("Loading models …\n")
    models: List[NSFWModel] = []

    model_builders = {
        "falconsai": FalconsaiModel,
        "marqo": MarqoModel,
        "siglip": SigLIPModel,
    }

    for key in model_keys:
        try:
            models.append(model_builders[key](device))
        except Exception as exc:
            print(f"\n  [ERROR] Failed to load '{key}' model: {exc}")
            print(f"           Skipping this model.\n")

    if not models:
        print("No models loaded successfully. Exiting.")
        sys.exit(1)

    model_names = [m.NAME for m in models]
    print(f"\nModels ready: {', '.join(model_names)}\n")

    # ---- Score every image with every model ---------------------------------
    results: List[Dict] = []
    timings: Dict[str, float] = {m.NAME: 0.0 for m in models}

    for path in tqdm(image_paths, desc="Benchmarking", unit="img"):
        img = load_image(path)
        if img is None:
            continue

        row: Dict = {"file": path, "filename": os.path.basename(path)}
        for model in models:
            t0 = time.perf_counter()
            try:
                s = model.score(img)
            except Exception as exc:
                print(f"\n  [WARN] {model.NAME} failed on {path}: {exc}")
                s = -1.0
            elapsed = time.perf_counter() - t0
            timings[model.NAME] += elapsed
            row[f"{model.NAME}_score"] = round(s, 4) if s >= 0 else "ERROR"
            row[f"{model.NAME}_flag"] = flag_label(s, threshold) if s >= 0 else "ERROR"
        results.append(row)

    # ---- Print comparison table ---------------------------------------------
    print("\n" + "=" * 100)
    print("  BENCHMARK RESULTS")
    print("=" * 100)
    print(f"  Threshold: {threshold}")
    print(f"  Device:    {device}")
    print("=" * 100)

    # Header
    name_col_w = 40
    score_col_w = 18
    header = f"  {'File':<{name_col_w}}"
    for mn in model_names:
        header += f"  {mn:^{score_col_w}}"
    print(header)
    print("  " + "-" * (name_col_w + (score_col_w + 2) * len(model_names)))

    for row in results:
        fname = row["filename"]
        if len(fname) > name_col_w - 2:
            fname = "…" + fname[-(name_col_w - 3) :]
        line = f"  {fname:<{name_col_w}}"
        for mn in model_names:
            score_val = row[f"{mn}_score"]
            flag_val = row[f"{mn}_flag"]
            if isinstance(score_val, float):
                cell = f"{score_val:.2%} {flag_val:>10}"
            else:
                cell = f"{'ERROR':>18}"
            line += f"  {cell:^{score_col_w}}"
        print(line)

    # ---- Agreement & disagreement summary -----------------------------------
    if len(models) >= 2:
        print("\n" + "-" * 100)
        print("  AGREEMENT ANALYSIS")
        print("-" * 100)

        agree_count = 0
        disagree_rows: List[Dict] = []
        for row in results:
            flags = []
            for mn in model_names:
                f = row.get(f"{mn}_flag", "ERROR")
                flags.append("NSFW" if f == "NSFW" else "safe")
            if len(set(flags)) == 1:
                agree_count += 1
            else:
                disagree_rows.append(row)

        total = len(results)
        print(f"  All models agree: {agree_count}/{total}  ({agree_count/total:.0%})")
        print(f"  Disagreements:    {len(disagree_rows)}/{total}")

        if disagree_rows:
            print(f"\n  Files where models disagree (inspect these closely):\n")
            for row in disagree_rows:
                parts = [f"    {row['filename']:<{name_col_w}}"]
                for mn in model_names:
                    sv = row[f"{mn}_score"]
                    fv = row[f"{mn}_flag"]
                    if isinstance(sv, float):
                        parts.append(f"{mn}: {sv:.2%} ({fv})")
                    else:
                        parts.append(f"{mn}: ERROR")
                print("  |  ".join(parts))

    # ---- Timing summary -----------------------------------------------------
    print("\n" + "-" * 100)
    print("  TIMING")
    print("-" * 100)
    total_images = len(results)
    for mn in model_names:
        total_t = timings[mn]
        avg_t = total_t / total_images if total_images else 0
        print(f"  {mn:<30}  total: {total_t:6.1f}s   avg: {avg_t*1000:6.1f}ms/image")

    # ---- Flagged count per model --------------------------------------------
    print("\n" + "-" * 100)
    print("  FLAGGED COUNTS")
    print("-" * 100)
    for mn in model_names:
        nsfw_count = sum(
            1 for r in results if r.get(f"{mn}_flag") == "NSFW"
        )
        print(f"  {mn:<30}  flagged: {nsfw_count}/{total_images}")
    print("=" * 100)

    # ---- Write CSV ----------------------------------------------------------
    if results:
        fieldnames = ["filename", "file"]
        for mn in model_names:
            fieldnames.extend([f"{mn}_score", f"{mn}_flag"])

        with open(output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {output_csv}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark three NSFW models side-by-side on a folder of test images.",
    )
    p.add_argument(
        "image_folder",
        help="Folder containing test images.",
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.75,
        help="NSFW probability threshold (0.0–1.0). Default: 0.75.",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV path. Default: benchmark_results.csv.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    p.add_argument(
        "--models", "-m",
        nargs="+",
        choices=sorted(MODEL_CHOICES),
        default=sorted(MODEL_CHOICES),
        help="Which models to benchmark. Default: all three. "
             "Options: falconsai, marqo, siglip",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    folder = os.path.abspath(args.image_folder)
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3) if hasattr(
            torch.cuda.get_device_properties(0), "total_mem"
        ) else torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {gpu} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("CUDA not available — using CPU.")

    run_benchmark(
        image_folder=folder,
        model_keys=args.models,
        threshold=args.threshold,
        device=device,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
