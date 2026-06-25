"""
Rule-based (scripted) preference labeling from episode metadata.

Importable interface:
    generate_scripted_pairs(registry_csv, output_csv, max_pairs) -> int

Label rules (applied in priority order):
    1. success_label: success beats failure
    2. collision (if present in metadata): no collision beats collision
    3. distance_to_goal (if present in metadata): lower is better
    4. Tie on all metrics → 0.5
"""

import csv
from itertools import combinations
from pathlib import Path


def _score_rollout(row: dict) -> tuple:
    """Return a comparable tuple; higher = better episode."""
    success = 1 if str(row.get("success_label", "false")).lower() == "true" else 0
    no_collision = 1 if str(row.get("collision", "false")).lower() == "false" else 0
    # Lower distance_to_goal is better; negate so higher tuple = better
    try:
        dist_score = -float(row.get("distance_to_goal", 0))
    except (ValueError, TypeError):
        dist_score = 0.0
    return (success, no_collision, dist_score)


def generate_scripted_pairs(
    registry_csv: str,
    output_csv: str,
    max_pairs: int | None = None,
) -> int:
    """
    Generate pairwise preference labels from rollout metadata fields.

    All (A, B) combinations are considered. Label:
        1.0  - A is strictly better than B
        0.0  - B is strictly better than A
        0.5  - equal on all available metrics

    Returns number of pairs written.
    """
    registry_csv = Path(registry_csv)
    output_csv = Path(output_csv)

    with open(registry_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        print(f"Need at least 2 rollouts to generate pairs; found {len(rows)}.")
        return 0

    pairs = list(combinations(rows, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["traj_A", "traj_B", "label"])
        writer.writeheader()

        for a, b in pairs:
            score_a = _score_rollout(a)
            score_b = _score_rollout(b)

            if score_a > score_b:
                label = 1.0
            elif score_b > score_a:
                label = 0.0
            else:
                label = 0.5

            writer.writerow({
                "traj_A": a["rollout_id"],
                "traj_B": b["rollout_id"],
                "label": label,
            })
            count += 1

    print(f"Wrote {count} scripted pairs to {output_csv}")
    return count
