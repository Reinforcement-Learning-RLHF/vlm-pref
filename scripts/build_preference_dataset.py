"""
Merge scripted_pairs.csv, vlm_pairs.csv, and human_pairs.csv into one
unified data/preferences/preference_dataset.csv.

Missing metadata columns (evaluator_type, timestamp, model_version,
prompt_version) are filled in with fallbacks so older CSVs without those
columns are handled gracefully.
"""

import csv
import sys
from datetime import datetime, timezone
from itertools import combinations
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFS_DIR = PROJECT_ROOT / "data" / "preferences"
REGISTRY_CSV = PROJECT_ROOT / "data" / "rollout_registry.csv"
OUTPUT_CSV = PREFS_DIR / "preference_dataset.csv"

FIELDNAMES = ["traj_A", "traj_B", "label", "evaluator_type", "timestamp", "model_version", "prompt_version"]

SOURCES = [
    ("scripted_pairs.csv", "scripted"),
    ("vlm_pairs.csv",      "vlm_gemini"),
    ("human_pairs.csv",    "human"),
]


def _file_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _expected_counts() -> dict[str, int] | None:
    """
    Read the registry and return expected pair counts per evaluator.
    Returns None if the registry is not available.
    """
    if not REGISTRY_CSV.exists():
        return None
    with open(REGISTRY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    successes = sum(1 for r in rows if str(r.get("success_label", "")).lower() == "true")
    failures = n - successes
    return {
        "scripted":   comb(n, 2),          # all combinations
        "vlm_gemini": successes * failures,  # success vs failure only
        "human":      successes * failures,
    }


def main() -> int:
    expected = _expected_counts()
    all_rows: list[dict] = []
    counts: dict[str, int] = {}

    for filename, evaluator_type in SOURCES:
        path = PREFS_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename}: not found")
            counts[evaluator_type] = 0
            continue

        fallback_ts = _file_mtime_utc(path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            if not row.get("evaluator_type"):
                row["evaluator_type"] = evaluator_type
            if not row.get("timestamp"):
                row["timestamp"] = fallback_ts
            if not row.get("model_version"):
                row["model_version"] = "n/a"
            if not row.get("prompt_version"):
                row["prompt_version"] = "n/a"

        counts[evaluator_type] = len(rows)
        all_rows.extend(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # --- summary ---
    total = len(all_rows)
    col_w = max(len(str(total)), 2)

    def _fmt(evaluator: str) -> str:
        n = counts.get(evaluator, 0)
        if expected and evaluator in expected:
            e = expected[evaluator]
            suffix = f"  (partial -- {n}/{e} labeled so far)" if 0 < n < e else ""
            return f"{n:>{col_w}}{suffix}"
        return f"{n:>{col_w}}"

    print(f"Scripted pairs:  {_fmt('scripted')}")
    print(f"VLM pairs:       {_fmt('vlm_gemini')}")
    print(f"Human pairs:     {_fmt('human')}")
    print(f"Total:           {total:>{col_w}}")
    print(f"\nSaved to {OUTPUT_CSV}")
    return total


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
