"""
Upload all rollout videos to a HuggingFace dataset repo and update rollout_registry.csv
with a video_url column so the Streamlit app can fetch them from the cloud.

Usage:
    python scripts/upload_videos_to_hf.py --repo-id your-org/robot-rollout-videos
    python scripts/upload_videos_to_hf.py --repo-id your-org/robot-rollout-videos --private

The script skips rollouts whose video file does not exist locally.
After running, commit the updated rollout_registry.csv.

Credentials: set HF_TOKEN env var or run `huggingface-cli login` first.
"""

import argparse
import csv
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. watai/robot-rollout-videos")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (default: $HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    if not args.dry_run:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
        )
        print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    with open(REGISTRY_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Registry is empty.")
        return

    fieldnames = list(rows[0].keys())
    if "video_url" not in fieldnames:
        fieldnames.append("video_url")

    skipped = 0
    uploaded = 0

    for row in rows:
        video_path = ROOT / row["video_path"]
        dest = f"videos/{row['rollout_id']}.mp4"
        url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/{dest}"

        if not video_path.exists():
            print(f"  SKIP  {row['rollout_id']} -- video not found locally")
            row.setdefault("video_url", "")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  DRY   {row['rollout_id']} -> {dest}")
        else:
            print(f"  UP    {row['rollout_id']} ({video_path.stat().st_size // 1024} KB) -> {dest}")
            api.upload_file(
                path_or_fileobj=str(video_path),
                path_in_repo=dest,
                repo_id=args.repo_id,
                repo_type="dataset",
            )

        row["video_url"] = url
        uploaded += 1

    if not args.dry_run:
        with open(REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nUploaded {uploaded} video(s), skipped {skipped}.")
        print(f"Updated {REGISTRY_CSV.name} with video_url column.")
        print("Next step: git add data/rollout_registry.csv && git commit")
    else:
        print(f"\nDry run: {uploaded} would upload, {skipped} would skip.")


if __name__ == "__main__":
    main()
