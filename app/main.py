"""
Human preference labeling UI.
Run: streamlit run app/main.py
"""

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"
OUTPUT_CSV = ROOT / "data" / "preferences" / "human_pairs.csv"

FIELDNAMES = [
    "traj_A", "traj_B", "label",
    "evaluator_type", "timestamp", "model_version", "prompt_version",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(REGISTRY_CSV, encoding="cp1252")
    df["success_label"] = df["success_label"].astype(str).str.lower() == "true"
    return df


def load_labeled_set() -> set[tuple[str, str]]:
    """Return already-labeled pairs as an undirected set (both orderings stored)."""
    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        return set()
    labeled: set[tuple[str, str]] = set()
    with OUTPUT_CSV.open() as f:
        for row in csv.DictReader(f):
            a, b = row["traj_A"], row["traj_B"]
            labeled.add((a, b))
            labeled.add((b, a))
    return labeled


def compute_pairs(df: pd.DataFrame, labeled: set) -> list[tuple[dict, dict]]:
    successes = df[df["success_label"]].to_dict("records")
    failures = df[~df["success_label"]].to_dict("records")
    return [
        (s, f) for s in successes for f in failures
        if (s["rollout_id"], f["rollout_id"]) not in labeled
    ]


def save_label(traj_a: str, traj_b: str, label: float) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with OUTPUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "traj_A": traj_a,
            "traj_B": traj_b,
            "label": label,
            "evaluator_type": "human",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": "human",
            "prompt_version": "n/a",
        })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Preference Labeler", layout="wide", page_icon="🤖")
    st.title("Robot Rollout Preference Labeler")

    df = load_registry()
    labeled = load_labeled_set()
    pairs = compute_pairs(df, labeled)

    n_success = int(df["success_label"].sum())
    n_failure = int((~df["success_label"]).sum())
    total_possible = n_success * n_failure
    labeled_count = total_possible - len(pairs)

    progress = labeled_count / total_possible if total_possible > 0 else 0.0
    st.progress(progress, text=f"{labeled_count} / {total_possible} pairs labeled")

    if total_possible == 0:
        st.warning("Registry is empty or has no success/failure split. Add rollouts first.")
        return

    if not pairs:
        st.success("🎉 All pairs labeled! Head to the Results page for a summary.")
        return

    # Sample a new pair only on first load or after a label/skip action
    if "row_a" not in st.session_state:
        chosen = random.choice(pairs)
        st.session_state.row_a = chosen[0]
        st.session_state.row_b = chosen[1]

    row_a: dict = st.session_state.row_a
    row_b: dict = st.session_state.row_b

    # Task context
    st.subheader(f"Task: {row_a['task_name']}")
    st.caption(f"Goal: {row_a['task_goal']}")
    st.divider()

    # Side-by-side videos
    col_a, col_b = st.columns(2)

    def render_video(col, label: str, row: dict) -> None:
        with col:
            st.markdown(f"#### Rollout {label}")
            video_path = ROOT / row["video_path"]
            if video_path.exists():
                st.video(str(video_path))
            else:
                st.warning(f"Video not found: `{row['video_path']}`")
                st.caption("Add the video file and rerun.")
            with st.expander("Notes"):
                st.write(row.get("additional_notes") or "—")

    render_video(col_a, "A", row_a)
    render_video(col_b, "B", row_b)

    st.divider()

    # Preference buttons — centered trio
    _, btn_a, btn_tie, btn_b, _ = st.columns([2, 1, 1, 1, 2])

    def on_label(label: float) -> None:
        save_label(row_a["rollout_id"], row_b["rollout_id"], label)
        del st.session_state.row_a
        del st.session_state.row_b

    with btn_a:
        if st.button("👈  A is better", use_container_width=True, type="primary"):
            on_label(1.0)
            st.rerun()

    with btn_tie:
        if st.button("🤝  Tie", use_container_width=True):
            on_label(0.5)
            st.rerun()

    with btn_b:
        if st.button("B is better  👉", use_container_width=True, type="primary"):
            on_label(0.0)
            st.rerun()

    # Skip — no label written
    _, skip_col = st.columns([5, 1])
    with skip_col:
        if st.button("Skip →", use_container_width=True, help="Skip without labeling"):
            del st.session_state.row_a
            del st.session_state.row_b
            st.rerun()


if __name__ == "__main__":
    main()
