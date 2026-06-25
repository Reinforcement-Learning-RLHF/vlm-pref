"""
One-off conversion script: data/test/ -> data/rollouts/ registry format.

Reads data/test/metadata.json, creates one subfolder per episode under
data/rollouts/, writes a metadata.json per episode, and copies the
agentview mp4 into place.

Usage:
    python scripts/convert_test_data.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "test"
ROLLOUTS_DIR = PROJECT_ROOT / "data" / "rollouts"


def main():
    source_meta = TEST_DIR / "metadata.json"
    if not source_meta.exists():
        print(f"ERROR: {source_meta} not found.")
        sys.exit(1)

    episodes = json.loads(source_meta.read_text(encoding="utf-8"))

    created = []
    skipped = []

    for ep in episodes:
        name = ep["name"]
        rollout_dir = ROLLOUTS_DIR / name
        rollout_dir.mkdir(parents=True, exist_ok=True)

        # Source video: metadata stores a Windows-style relative path with a
        # pour_videos\ prefix, but the files are flat in data/test/.
        src_video_filename = f"{name}_agentview.mp4"
        src_video = TEST_DIR / src_video_filename
        dst_video = rollout_dir / src_video_filename

        if not src_video.exists():
            print(f"  [skip] {name}: source video not found at {src_video}")
            skipped.append(name)
            continue

        # Build target metadata
        metadata = {
            "rollout_id": name,
            "task_name": "pour",
            "task_goal": "Pour the contents of the cup into the target container.",
            "video_filename": src_video_filename,
            "video_path": f"data/rollouts/{name}/{src_video_filename}",
            "simulator": "custom",
            "success_label": ep["label"] == "good",
            "timestamp": "2026-03-26T00:00:00Z",
            "additional_notes": ep["description"],
            "total_reward": ep["stats"]["total_reward"],
            "steps_recorded": ep["stats"]["steps_recorded"],
            "tilt_deg": ep["params"]["tilt_deg"],
            "noise": ep["params"]["noise"],
        }

        (rollout_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        shutil.copy2(src_video, dst_video)
        created.append(name)
        print(f"  [ok] {name}  (label={ep['label']}, reward={ep['stats']['total_reward']})")

    print(f"\nDone: {len(created)} rollout(s) created, {len(skipped)} skipped.")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
