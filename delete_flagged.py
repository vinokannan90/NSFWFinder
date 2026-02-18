"""
Delete Flagged NSFW Files
=========================
Reads the review_list.csv and deletes original files that have been
flagged as "DELETE".  Also removes the copied review files.

Usage:
    python delete_flagged.py [options]

Examples:
    python delete_flagged.py --review FindingsToReview\\review_list.csv
    python delete_flagged.py --review FindingsToReview\\review_list.csv --dry-run
"""

import argparse
import csv
import os
import sys

try:
    from send2trash import send2trash
except ImportError:
    print("Error: 'send2trash' package is required.")
    print("Install it with:  pip install send2trash")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete original files flagged as DELETE in the review CSV.",
    )
    parser.add_argument(
        "--review", "-r",
        type=str,
        default=os.path.join("FindingsToReview", "review_list.csv"),
        help="Path to the review_list.csv. Default: FindingsToReview\\review_list.csv",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Preview what would be deleted without actually deleting.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Also delete the FindingsToReview folder after deletion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review_path = os.path.abspath(args.review)
    review_dir = os.path.dirname(review_path)

    if not os.path.isfile(review_path):
        print(f"Error: Review file not found: {review_path}")
        sys.exit(1)

    # Read review list
    entries: list[dict] = []
    with open(review_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entries.append(row)

    # Categorize
    to_delete = [e for e in entries if e.get("Flag", "").strip().upper() == "DELETE"]
    to_keep = [e for e in entries if e.get("Flag", "").strip().upper() == "KEEP"]
    to_review = [e for e in entries if e.get("Flag", "").strip().upper() not in ("DELETE", "KEEP")]

    print("=" * 60)
    print("  DELETE FLAGGED FILES — SUMMARY")
    print("=" * 60)
    print(f"  Total entries   : {len(entries)}")
    print(f"  Flagged DELETE  : {len(to_delete)}")
    print(f"  Flagged KEEP    : {len(to_keep)}")
    print(f"  Still REVIEW    : {len(to_review)}")
    print("=" * 60)

    if to_review:
        print(f"\n  WARNING: {len(to_review)} file(s) still marked as REVIEW.")
        print("  These will NOT be deleted. Review them first.\n")

    if not to_delete:
        print("\n  No files flagged for deletion. Nothing to do.")
        return

    if args.dry_run:
        print("\n  [DRY RUN] The following files WOULD be moved to Recycle Bin:\n")
    else:
        print(f"\n  About to move {len(to_delete)} original file(s) to the Recycle Bin.")
        confirm = input("  Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            print("  Aborted. No files were deleted.")
            return
        print()

    deleted = 0
    errors = 0
    trashed_originals: set[str] = set()

    for entry in to_delete:
        original = entry.get("Original Path", "")
        review_file = entry.get("Review File", "")
        review_file_path = os.path.join(review_dir, review_file) if review_file else ""

        if args.dry_run:
            status = "EXISTS" if os.path.isfile(original) else "NOT FOUND"
            print(f"    [{status}]  {original}")
            continue

        # Move original file to Recycle Bin
        if os.path.isfile(original):
            try:
                send2trash(original)
                print(f"    [RECYCLED] {original}")
                deleted += 1
                trashed_originals.add(original)
            except Exception as exc:
                print(f"    [ERROR]    {original}  —  {exc}")
                errors += 1
        else:
            print(f"    [SKIP]     {original}  —  File not found")

        # Delete the review copy too
        if review_file_path and os.path.isfile(review_file_path):
            try:
                os.remove(review_file_path)
            except Exception:
                pass  # Non-critical

    if not args.dry_run:
        print(f"\n  Recycled: {deleted}  |  Errors: {errors}")

        # Update review_list.csv — change Flag from DELETE to TRASHED
        if trashed_originals:
            for entry in entries:
                if entry.get("Original Path", "") in trashed_originals:
                    entry["Flag"] = "TRASHED"

            fieldnames = list(entries[0].keys())
            with open(review_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(entries)
            print(f"  Updated {len(trashed_originals)} entries to TRASHED in review_list.csv.")

        # Optionally clean up the review folder
        if args.cleanup and os.path.isdir(review_dir):
            remaining = [f for f in os.listdir(review_dir)
                         if f != "review_list.csv"]
            if not remaining:
                import shutil
                shutil.rmtree(review_dir)
                print(f"  Cleaned up review folder: {review_dir}")
            else:
                print(f"  Review folder still has {len(remaining)} file(s), not removed.")

    print()


if __name__ == "__main__":
    main()
