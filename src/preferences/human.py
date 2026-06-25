"""
Human preference labeling via interactive terminal session.

Importable interface:
    label_pair(video_path_1, video_path_2) -> float | None
    generate_human_pairs(registry_csv, output_csv, sample_size, seed) -> int
"""

import csv
import random
import subprocess
import sys
from pathlib import Path


def _open_video(path: str) -> None:
    """Open a video in the system default player."""
    path = Path(path)
    if not path.exists():
        print(f"  [warn] Video not found: {path}")
        return
    if sys.platform == "win32":
        subprocess.Popen(["start", "", str(path)], shell=True)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def label_pair(video_path_1: str, video_path_2: str) -> float | None:
    """
    Show two video paths, optionally open them, prompt for a preference.

    Returns:
        1.0   — Video A is better
        0.0   — Video B is better
        0.5   — Tie
        None  — Skipped (not written to CSV)
    """
    print(f"\n  Video A: {video_path_1}")
    print(f"  Video B: {video_path_2}")

    open_choice = input("  Open videos in player? [y/N]: ").strip().lower()
    if open_choice == "y":
        _open_video(video_path_1)
        _open_video(video_path_2)

    while True:
        choice = input("  Which is better? [1=A  2=B  0=tie  s=skip]: ").strip().lower()
        if choice == "1":
            return 1.0
        elif choice == "2":
            return 0.0
        elif choice == "0":
            return 0.5
        elif choice == "s":
            return None
        else:
            print("  Invalid input. Enter 1, 2, 0, or s.")


def generate_human_pairs(
    registry_csv: str,
    output_csv: str,
    sample_size: int | None = None,
    seed: int | None = None,
) -> int:
    """
    Interactively collect human preference labels for video pairs.

    Pairs successes against failures. If sample_size is given, randomly samples
    that many pairs. Appends to output_csv so interrupted sessions don't lose progress.

    Returns number of pairs labeled (skips not counted).
    """
    registry_csv = Path(registry_csv)
    output_csv = Path(output_csv)

    with open(registry_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    successes = [r for r in rows if str(r.get("success_label", "false")).lower() == "true"]
    failures = [r for r in rows if str(r.get("success_label", "false")).lower() == "false"]
    pairs = [(s, fail) for s in successes for fail in failures]

    if seed is not None:
        random.seed(seed)
    random.shuffle(pairs)

    if sample_size is not None:
        pairs = pairs[:sample_size]

    print(f"\nHuman labeling session: {len(pairs)} pair(s) to label.")
    print("Keys: 1=A is better  2=B is better  0=tie  s=skip\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["traj_A", "traj_B", "label"])
        if write_header:
            writer.writeheader()

        count = 0
        for i, (a, b) in enumerate(pairs):
            print(f"[{i + 1}/{len(pairs)}]  {a['rollout_id']}  vs  {b['rollout_id']}")
            result = label_pair(a["video_path"], b["video_path"])
            if result is not None:
                writer.writerow({
                    "traj_A": a["rollout_id"],
                    "traj_B": b["rollout_id"],
                    "label": result,
                })
                count += 1

    print(f"\nLabeled {count} pair(s). Results saved to {output_csv}")
    return count
