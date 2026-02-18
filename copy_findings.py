"""
Copy Findings to Review
=======================
Reads the NSFW scan report CSV and copies all flagged files into a
"FindingsToReview" folder for manual review.  Also generates a
"review_list.csv" with a Flag column for marking items to delete.

Usage:
    python copy_findings.py [options]

Examples:
    python copy_findings.py
    python copy_findings.py --report nsfw_report.csv --output D:\\FindingsToReview
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy NSFW-flagged files to a review folder.",
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        default="nsfw_report.csv",
        help="Path to the NSFW scan report CSV. Default: nsfw_report.csv",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="FindingsToReview",
        help="Output folder for copied files. Default: FindingsToReview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_path = os.path.abspath(args.report)
    output_dir = os.path.abspath(args.output)

    if not os.path.isfile(report_path):
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)

    # Read the report
    entries: list[dict] = []
    with open(report_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entries.append(row)

    if not entries:
        print("No entries found in the report. Nothing to copy.")
        return

    print(f"Found {len(entries)} flagged files in report.")
    print(f"Copying to: {output_dir}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Copy files — use a numbered prefix to avoid name collisions
    # while keeping the original filename readable
    review_rows: list[dict] = []
    copied = 0
    skipped = 0

    for idx, entry in enumerate(entries, start=1):
        src_path = entry["File Path"]
        file_type = entry.get("Type", "")
        nsfw_score = entry.get("NSFW Score", "")
        frame_scores = entry.get("Frame Scores", "")

        if not os.path.isfile(src_path):
            print(f"  [SKIP] File not found: {src_path}")
            skipped += 1
            continue

        # Create a unique destination filename: 001_originalname.ext
        original_name = os.path.basename(src_path)
        dest_name = f"{idx:04d}_{original_name}"
        dest_path = os.path.join(output_dir, dest_name)

        # Handle unlikely collision
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(dest_name)
            dest_name = f"{name}_dup{ext}"
            dest_path = os.path.join(output_dir, dest_name)

        shutil.copy2(src_path, dest_path)
        copied += 1

        review_rows.append({
            "No": idx,
            "Review File": dest_name,
            "Original Path": src_path,
            "Type": file_type,
            "NSFW Score": nsfw_score,
            "Frame Scores": frame_scores,
            "Flag": "REVIEW",
        })

    # Write review_list.csv
    review_csv_path = os.path.join(output_dir, "review_list.csv")
    with open(review_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "No", "Review File", "Original Path", "Type",
            "NSFW Score", "Frame Scores", "Flag",
        ])
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"\nDone!  Copied: {copied}  |  Skipped: {skipped}")
    print(f"\nReview list saved to: {review_csv_path}")
    print()
    print("=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print()
    print("  1. Open the review folder to browse the files:")
    print(f"     {output_dir}")
    print()
    print("  2. Open review_list.csv in Excel and update the")
    print("     'Flag' column for each file:")
    print()
    print("       DELETE  — Confirmed NSFW, delete the original")
    print("       KEEP    — False positive, keep the original")
    print("       REVIEW  — Not yet reviewed (default)")
    print()
    print("  3. Save the CSV, then run:")
    print(f'     python delete_flagged.py --review "{review_csv_path}"')
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
