# NSFWFinder

A local, GPU-accelerated tool that scans a drive or folder for NSFW images and videos so that you can safeguard any drive you possess or acquire. It uses a pre-trained Vision Transformer model ([Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)) to classify files — no data ever leaves your machine.

The full workflow covers **scanning**, **reviewing**, and **deleting**:

1. **`nsfw_scanner.py`** — Scans a drive/folder and generates a report of flagged files
2. **`copy_findings.py`** — Copies flagged files into a review folder for easy browsing
3. **`delete_flagged.py`** — Moves confirmed NSFW files to the Recycle Bin after your review

---

## Minimum System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Any modern quad-core | AMD Ryzen 5000+ / Intel 12th Gen+ |
| **GPU** | NVIDIA GPU with 4 GB+ VRAM | NVIDIA RTX 3060+ (8 GB+ VRAM) |
| **GPU Driver** | NVIDIA driver 551.78+ (Windows) / 550.54.14+ (Linux) | Latest via [NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app/) or [GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | ~500 MB free (model + packages) | SSD for faster file reads |
| **OS** | Windows 10/11, Linux | Windows 10/11 |

> **Note:** The scanner can run on CPU-only machines but will be significantly slower. An NVIDIA GPU is strongly recommended.
>
> **You do NOT need to install CUDA, cuDNN, or the CUDA Toolkit separately.** Everything is bundled automatically when you install the Python dependencies.

---

## Setup

### 1. Install Python (if you don't have it)

Download and install **Python 3.12** (or newer) from the official site:

> **https://www.python.org/downloads/**

**Important:** During installation, check **"Add Python to PATH"** — this is required.

To verify Python is installed, open a terminal and run:

```bash
python --version
```

You should see something like `Python 3.12.x`.

---

### 2. Clone the repository

```bash
git clone https://github.com/vinokannan90/NSFWFinder.git
cd NSFWFinder
```

Or download the ZIP from GitHub and extract it.

---

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  venv\Scripts\activate
  ```
- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

---

### 4. Install all dependencies (one command)

```bash
pip install -r requirements.txt
```

This single command installs **everything** you need:

| Package | Purpose |
|---------|---------|
| **PyTorch + CUDA 12.4** | GPU-accelerated deep learning (includes CUDA runtime & cuDNN) |
| **Torchvision** | Image preprocessing & transforms |
| **Transformers** | Hugging Face model loading |
| **Pillow** | Image file handling |
| **OpenCV** | Video frame extraction |
| **tqdm** | Progress bars |
| **send2trash** | Recycle Bin support for safe deletion |

> **No separate CUDA, cuDNN, or CUDA Toolkit installation is needed.** The PyTorch CUDA wheel bundles everything.
>
> The first run will also download the classification model (~350 MB) from Hugging Face. After that, everything runs fully offline.

---

## Usage

### Step 1 — Scan a drive or folder

```bash
python nsfw_scanner.py <path_to_scan>
```

**Examples:**

```bash
# Scan entire D: drive
python nsfw_scanner.py D:\

# Scan a specific folder
python nsfw_scanner.py "C:\Users\Photos"

# Custom threshold and batch size
python nsfw_scanner.py D:\ --threshold 0.8 --batch-size 32

# Force CPU mode (no GPU)
python nsfw_scanner.py D:\ --cpu

# Custom batch size io workers and video workers
python nsfw_scanner.py F:\ --batch-size 48 --io-workers 12 --video-workers 6
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold, -t` | `0.75` | NSFW score threshold (0.0–1.0) |
| `--batch-size, -b` | `32` | GPU batch size (lower if you get VRAM errors) |
| `--video-frames, -vf` | `8` | Frames sampled per video |
| `--report, -r` | `nsfw_report.csv` | Output report file path |
| `--io-workers` | `8` | I/O threads for loading files |
| `--video-workers` | `4` | Threads for video decoding |
| `--cpu` | off | Force CPU-only mode |
| `--verbose, -v` | off | Enable debug logging |

**Output:** A `nsfw_report.csv` file listing all flagged files with their NSFW scores.

---

### Step 2 — Copy findings for review

```bash
python copy_findings.py
```

This reads `nsfw_report.csv` and:
- Copies all flagged files into a `FindingsToReview` folder
- Generates a `review_list.csv` with a **Flag** column

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--report, -r` | `nsfw_report.csv` | Input report path |
| `--output, -o` | `FindingsToReview` | Output folder |

---

### Step 3 — Review and flag files

Open `FindingsToReview\review_list.csv` in Excel or any spreadsheet editor. For each file, update the **Flag** column:

| Flag Value | Meaning |
|------------|---------|
| `DELETE` | Confirmed NSFW — will be moved to Recycle Bin |
| `KEEP` | False positive — will not be touched |
| `REVIEW` | Not yet reviewed (default) |

Browse the copied files in the `FindingsToReview` folder to verify each one.

---

### Step 4 — Delete confirmed NSFW files

Preview what would be deleted (dry run):

```bash
python delete_flagged.py --dry-run
```

Delete for real (moves originals to Recycle Bin):

```bash
python delete_flagged.py
```

You will be asked to type `YES` to confirm. Files are moved to the **Recycle Bin**, not permanently deleted — you can restore them if needed.

After deletion, the `review_list.csv` is updated automatically: `DELETE` → `TRASHED`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--review, -r` | `FindingsToReview\review_list.csv` | Review CSV path |
| `--dry-run, -d` | off | Preview without deleting |
| `--cleanup` | off | Remove the review folder after deletion |

---

## Supported File Types

**Images:** `.jpg` `.jpeg` `.png` `.bmp` `.gif` `.tiff` `.tif` `.webp`

**Videos:** `.mp4` `.avi` `.mkv` `.mov` `.wmv` `.flv` `.webm` `.mpeg` `.mpg` `.m4v` `.3gp`

---

## How It Works

1. **Discovery** — Recursively walks the target path collecting image and video files.
2. **Pipelined Processing** — Multiple I/O threads load files from disk in parallel while the GPU processes the previous batch, keeping the GPU constantly fed.
3. **Image Classification** — Images are classified in GPU batches using a Vision Transformer model in FP16 half-precision.
4. **Video Classification** — Evenly-spaced frames are extracted from each video and classified. The **average** frame score determines whether the video is flagged.
5. **Reporting** — Flagged files are written to a CSV report sorted by score.

---

## Batch Size Guide

| GPU VRAM | Recommended `--batch-size` |
|----------|---------------------------|
| 4 GB | 8 |
| 6 GB | 16 |
| 8 GB | 24 |
| 12 GB | 32–48 |
| 16 GB+ | 64 |

If you get a `CUDA OutOfMemoryError`, reduce the batch size.

---

## I/O Workers Guide

`--io-workers` controls how many threads load images from disk in parallel, feeding the GPU.

| CPU Cores | Drive Type | Recommended `--io-workers` |
|-----------|------------|---------------------------|
| 4 | HDD | 4 |
| 4 | SSD | 6 |
| 6–8 | HDD | 6 |
| 6–8 | SSD | 8–10 |
| 12+ | HDD | 8 |
| 12+ | SSD | 10–16 |

More workers help when the GPU is waiting on file reads. If CPU usage is already near 100%, adding more workers won't help. On HDDs, going too high can cause seek thrashing — keep it moderate.

---

## Video Workers Guide

`--video-workers` controls how many threads decode video frames in parallel.

| CPU Cores | Recommended `--video-workers` |
|-----------|-------------------------------|
| 4 | 2 |
| 6–8 | 4 |
| 12+ | 6–8 |

Video decoding is CPU-intensive. Set this lower than `--io-workers` to leave headroom for the rest of the pipeline. If you're scanning a path with very few videos, this setting has minimal impact.

---

## Video Frames Guide

`--video-frames` controls how many evenly-spaced frames are sampled from each video for classification. The average score across all sampled frames determines whether the video is flagged.

| Use Case | Recommended `--video-frames` | Trade-off |
|----------|------------------------------|-----------|
| Quick scan | 4 | Faster, but may miss short scenes |
| Balanced (default) | 8 | Good coverage for most content |
| Thorough scan | 16 | Better detection, ~2× slower per video |
| Maximum accuracy | 24–32 | Best for short clips, significantly slower |

**Tips:**
- For long movies / TV episodes, 8 frames is usually sufficient since NSFW content in these tends to span multiple scenes.
- For short clips (under 2 minutes), use 16+ frames for better sampling density.
- Doubling the frame count roughly doubles the processing time per video.

---

## Privacy

- All processing happens **locally** on your machine.
- No files or data are uploaded anywhere.
- The only network call is a **one-time model download** (~350 MB) from Hugging Face on first run. After that, the model is cached and the tool works fully offline.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
