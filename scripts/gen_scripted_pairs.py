"""Generate scripted (rule-based) preference pairs from rollout metadata."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preferences.scripted import generate_scripted_pairs

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate scripted preference pairs.")
    parser.add_argument(
        "--registry", type=Path,
        default=project_root / "data" / "rollout_registry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "data" / "preferences" / "scripted_pairs.csv",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    generate_scripted_pairs(args.registry, args.output, args.max_pairs)
