"""
One-time script: add video_url column to rollout_registry.csv
pointing to the HF dataset repo where videos were uploaded.

Usage:
    python scripts/patch_registry_video_urls.py
    python scripts/patch_registry_video_urls.py --repo-id nafisatibrahim/vlm-pref-videos
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "rollout_registry.csv"
DEFAULT_REPO = "nafisatibrahim/vlm-pref-videos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    args = parser.parse_args()

    with open(REGISTRY, newline="", encoding="cp1252") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    if "video_url" not in fieldnames:
        fieldnames.append("video_url")

    base = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main"

    for row in rows:
        # video_path is like "data/rollouts/pour_success_1/agentview.mp4"
        # uploaded to HF as "rollouts/pour_success_1/agentview.mp4"
        video_path = row.get("video_path", "")
        # Strip the leading "data/" since upload_folder used path_in_repo="rollouts"
        hf_path = video_path.replace("data/", "", 1).replace("\\", "/")
        row["video_url"] = f"{base}/{hf_path}"

    with open(REGISTRY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} rows.")
    print(f"Sample URL: {rows[0]['video_url']}")
    print("Next: git add data/rollout_registry.csv && git commit -m 'add video_url column'")


if __name__ == "__main__":
    main()
