"""Collect human preference labels interactively for video pairs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preferences.human import generate_human_pairs

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Collect human preference labels for video pairs.")
    parser.add_argument(
        "--registry", type=Path,
        default=project_root / "data" / "rollout_registry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "data" / "preferences" / "human_pairs.csv",
    )
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of pairs to label (default: all success-vs-failure pairs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for pair shuffling")
    args = parser.parse_args()
    generate_human_pairs(args.registry, args.output, args.sample_size, args.seed)
