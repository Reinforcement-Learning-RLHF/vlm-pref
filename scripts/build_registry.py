"""Build data/rollout_registry.csv from data/rollouts/."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.registry import build_registry

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build rollout_registry.csv from rollout folders.")
    parser.add_argument(
        "--rollouts-dir", type=Path,
        default=project_root / "data" / "rollouts",
        help="Path to rollouts folder (default: data/rollouts/)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "data" / "rollout_registry.csv",
        help="Output CSV path (default: data/rollout_registry.csv)",
    )
    args = parser.parse_args()
    build_registry(args.rollouts_dir, args.output)
