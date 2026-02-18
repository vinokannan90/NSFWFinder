"""
NSFW File Scanner
=================
Scans a specified drive or directory for NSFW images and videos using a
pre-trained Vision Transformer model.  Accelerated with NVIDIA CUDA (tested
on RTX 4070 Super).

Uses a pipelined architecture:
  - Multiple I/O threads read & decode files from disk
  - A prefetch queue feeds ready batches to the GPU
  - The GPU processes batches at full utilisation

Usage:
    python nsfw_scanner.py <path_to_scan> [options]

Examples:
    python nsfw_scanner.py D:\\
    python nsfw_scanner.py "C:\\Users\\Photos" --threshold 0.7 --batch-size 32
    python nsfw_scanner.py E:\\ --video-frames 5 --report results.csv
"""

import argparse
import csv
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageClassification, ViTImageProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg",
    ".mpg", ".m4v", ".3gp",
}

MODEL_NAME = "Falconsai/nsfw_image_detection"

# Pipeline constants
IO_WORKERS = 8          # threads for reading files from disk
PREFETCH_BATCHES = 4    # number of batches to prefetch ahead of GPU
VIDEO_DECODE_WORKERS = 4  # threads for parallel video decoding

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    file_path: str
    file_type: str          # "image" | "video"
    nsfw_score: float       # 0.0 – 1.0
    is_nsfw: bool
    frame_scores: List[float] = field(default_factory=list)


# Sentinel value to signal end of queue
_SENTINEL = None


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class NSFWClassifier:
    """Wraps the Falconsai/nsfw_image_detection ViT model."""

    def __init__(self, device: str = "cuda", half_precision: bool = True):
        logging.info("Loading NSFW classification model '%s' …", MODEL_NAME)
        self.device = torch.device(device)
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

        self._use_half = half_precision and self.device.type == "cuda"
        if self._use_half:
            self.model = self.model.half()
            logging.info("Using FP16 (half-precision) for faster inference.")

        self.model.to(self.device)
        self._nsfw_index = int(self.model.config.label2id.get("nsfw", 1))
        logging.info("Model loaded on %s.", self.device)

    @torch.inference_mode()
    def classify_images(self, images: List[Image.Image]) -> List[float]:
        """Return a list of NSFW probability scores for a batch of PIL Images."""
        if not images:
            return []

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self._use_half:
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self._nsfw_index].cpu().tolist()

    @torch.inference_mode()
    def classify_preprocessed(self, pixel_values: torch.Tensor) -> List[float]:
        """Classify a pre-processed batch tensor (already on device)."""
        if self._use_half:
            pixel_values = pixel_values.half()

        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self._nsfw_index].cpu().tolist()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(root: str) -> tuple[list[str], list[str]]:
    """Walk *root* and return (image_files, video_files)."""
    images: list[str] = []
    videos: list[str] = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            full = os.path.join(dirpath, fname)
            if ext in IMAGE_EXTENSIONS:
                images.append(full)
            elif ext in VIDEO_EXTENSIONS:
                videos.append(full)

    return images, videos


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        # Force decode by accessing pixel data (avoids lazy-load stalls later)
        img.load()
        return img
    except Exception as exc:
        logging.debug("Skipped %s: %s", path, exc)
        return None


def _load_image_with_path(path: str) -> Tuple[str, Optional[Image.Image]]:
    return path, _load_image(path)


# ---------------------------------------------------------------------------
# Pipelined image scanning
# ---------------------------------------------------------------------------

def _image_producer(
    image_paths: list[str],
    batch_size: int,
    out_queue: queue.Queue,
):
    """I/O thread: load images in parallel and push ready batches into queue."""
    batch_paths: list[str] = []
    batch_images: list[Image.Image] = []

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        futures = {
            pool.submit(_load_image_with_path, p): p
            for p in image_paths
        }

        for future in as_completed(futures):
            path, img = future.result()
            if img is not None:
                batch_paths.append(path)
                batch_images.append(img)

                if len(batch_images) >= batch_size:
                    out_queue.put((list(batch_paths), list(batch_images)))
                    batch_paths.clear()
                    batch_images.clear()

    # Flush remaining
    if batch_images:
        out_queue.put((list(batch_paths), list(batch_images)))

    out_queue.put(_SENTINEL)


def scan_images(
    classifier: NSFWClassifier,
    image_paths: list[str],
    threshold: float,
    batch_size: int,
) -> list[ScanResult]:
    """Classify all images using a producer/consumer pipeline."""
    results: list[ScanResult] = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    # Prefetch queue — I/O threads fill it, GPU thread drains it
    prefetch_q: queue.Queue = queue.Queue(maxsize=PREFETCH_BATCHES)

    producer = threading.Thread(
        target=_image_producer,
        args=(image_paths, batch_size, prefetch_q),
        daemon=True,
    )
    producer.start()

    pbar = tqdm(total=len(image_paths), desc="Scanning images", unit="file")

    while True:
        item = prefetch_q.get()
        if item is _SENTINEL:
            break

        batch_paths, batch_images = item
        scores = classifier.classify_images(batch_images)

        for path, score in zip(batch_paths, scores):
            results.append(ScanResult(
                file_path=path,
                file_type="image",
                nsfw_score=round(score, 4),
                is_nsfw=score >= threshold,
            ))

        pbar.update(len(batch_paths))

        # Free memory explicitly
        del batch_images
        del scores

    pbar.close()
    producer.join()
    return results


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(video_path: str, num_frames: int) -> list[Image.Image]:
    """Extract *num_frames* evenly spaced frames from a video."""
    frames: list[Image.Image] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.debug("Cannot open video: %s", video_path)
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    step = max(total_frames // (num_frames + 1), 1)
    target_indices = [step * (i + 1) for i in range(num_frames)]

    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()
    return frames


def _extract_video_data(
    video_path: str, num_frames: int
) -> Tuple[str, list[Image.Image]]:
    """Extract frames for one video — used by thread pool."""
    return video_path, _extract_frames(video_path, num_frames)


# ---------------------------------------------------------------------------
# Pipelined video scanning
# ---------------------------------------------------------------------------

def _video_producer(
    video_paths: list[str],
    num_frames: int,
    out_queue: queue.Queue,
):
    """Decode videos in parallel threads and push frames into queue."""
    with ThreadPoolExecutor(max_workers=VIDEO_DECODE_WORKERS) as pool:
        futures = {
            pool.submit(_extract_video_data, vp, num_frames): vp
            for vp in video_paths
        }

        for future in as_completed(futures):
            vpath, frames = future.result()
            if frames:
                out_queue.put((vpath, frames))

    out_queue.put(_SENTINEL)


def scan_videos(
    classifier: NSFWClassifier,
    video_paths: list[str],
    threshold: float,
    num_frames: int,
    batch_size: int,
) -> list[ScanResult]:
    """Classify videos using a pipelined producer/consumer approach."""
    results: list[ScanResult] = []

    prefetch_q: queue.Queue = queue.Queue(maxsize=PREFETCH_BATCHES)

    producer = threading.Thread(
        target=_video_producer,
        args=(video_paths, num_frames, prefetch_q),
        daemon=True,
    )
    producer.start()

    pbar = tqdm(total=len(video_paths), desc="Scanning videos", unit="file")

    while True:
        item = prefetch_q.get()
        if item is _SENTINEL:
            break

        vpath, frames = item

        # Process frames in batches
        all_scores: list[float] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            all_scores.extend(classifier.classify_images(batch))

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        results.append(ScanResult(
            file_path=vpath,
            file_type="video",
            nsfw_score=round(avg_score, 4),
            is_nsfw=avg_score >= threshold,
            frame_scores=[round(s, 4) for s in all_scores],
        ))

        pbar.update(1)
        del frames
        del all_scores

    pbar.close()
    producer.join()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(results: list[ScanResult], report_path: str) -> None:
    """Write flagged results to a CSV file."""
    flagged = [r for r in results if r.is_nsfw]
    flagged.sort(key=lambda r: r.nsfw_score, reverse=True)

    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["File Path", "Type", "NSFW Score", "Frame Scores"])
        for r in flagged:
            frame_info = "; ".join(str(s) for s in r.frame_scores) if r.frame_scores else ""
            writer.writerow([r.file_path, r.file_type, r.nsfw_score, frame_info])

    logging.info("Report saved to %s  (%d flagged files).", report_path, len(flagged))


def print_summary(results: list[ScanResult], threshold: float) -> None:
    flagged = [r for r in results if r.is_nsfw]
    total_images = sum(1 for r in results if r.file_type == "image")
    total_videos = sum(1 for r in results if r.file_type == "video")
    flagged_images = sum(1 for r in flagged if r.file_type == "image")
    flagged_videos = sum(1 for r in flagged if r.file_type == "video")

    print("\n" + "=" * 60)
    print("  NSFW SCAN SUMMARY")
    print("=" * 60)
    print(f"  Threshold       : {threshold}")
    print(f"  Images scanned  : {total_images}  (flagged: {flagged_images})")
    print(f"  Videos scanned  : {total_videos}  (flagged: {flagged_videos})")
    print(f"  Total flagged   : {len(flagged)}")
    print("=" * 60)

    if flagged:
        print("\n  Flagged files (sorted by score, highest first):\n")
        flagged.sort(key=lambda r: r.nsfw_score, reverse=True)
        for r in flagged:
            tag = "[IMG]" if r.file_type == "image" else "[VID]"
            print(f"    {tag}  {r.nsfw_score:.2%}  {r.file_path}")
    else:
        print("\n  No NSFW files detected.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a drive or folder for NSFW images and videos (GPU-accelerated).",
    )
    parser.add_argument(
        "scan_path",
        help="Root path to scan (e.g. D:\\ or C:\\Users\\Photos).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.75,
        help="NSFW probability threshold (0.0–1.0). Default: 0.75.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for GPU inference. Default: 32.",
    )
    parser.add_argument(
        "--video-frames", "-vf",
        type=int,
        default=8,
        help="Number of frames to sample per video. Default: 8.",
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        default="nsfw_report.csv",
        help="Output CSV report file path. Default: nsfw_report.csv.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (ignore GPU).",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=IO_WORKERS,
        help=f"Number of I/O threads for loading files. Default: {IO_WORKERS}.",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=VIDEO_DECODE_WORKERS,
        help=f"Number of threads for video decoding. Default: {VIDEO_DECODE_WORKERS}.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Allow CLI to override worker counts
    global IO_WORKERS, VIDEO_DECODE_WORKERS
    IO_WORKERS = args.io_workers
    VIDEO_DECODE_WORKERS = args.video_workers

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    scan_root = os.path.abspath(args.scan_path)
    if not os.path.isdir(scan_root):
        logging.error("Path does not exist or is not a directory: %s", scan_root)
        sys.exit(1)

    # ---- Device selection ---------------------------------------------------
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logging.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, vram)
    else:
        device = "cpu"
        logging.warning("CUDA not available – falling back to CPU. Install "
                        "PyTorch with CUDA support for GPU acceleration.")

    # ---- Discover files -----------------------------------------------------
    logging.info("Discovering files in %s …", scan_root)
    image_paths, video_paths = discover_files(scan_root)
    logging.info("Found %d images and %d videos.", len(image_paths), len(video_paths))

    if not image_paths and not video_paths:
        print("No image or video files found. Nothing to scan.")
        return

    # ---- Load model ---------------------------------------------------------
    classifier = NSFWClassifier(device=device, half_precision=(device == "cuda"))

    # ---- Scan ---------------------------------------------------------------
    start = time.perf_counter()
    results: list[ScanResult] = []

    if image_paths:
        results.extend(scan_images(classifier, image_paths, args.threshold, args.batch_size))

    if video_paths:
        results.extend(scan_videos(
            classifier, video_paths, args.threshold,
            args.video_frames, args.batch_size,
        ))

    elapsed = time.perf_counter() - start
    logging.info("Scan completed in %.1f seconds.", elapsed)

    # ---- Report -------------------------------------------------------------
    print_summary(results, args.threshold)
    write_report(results, args.report)


if __name__ == "__main__":
    main()