# Pipeline Overview

This project compares three sources of preference feedback for training reward models on robot manipulation tasks (pick-and-place with a real SO-101 arm via LeRobot).

**Step 1 — Collect episodes.** The Robotics team records mp4 videos and writes a `metadata.json` per episode, depositing each into `data/rollouts/<rollout_id>/`. See [rollout_format.md](rollout_format.md) for the required fields.

**Step 2 — Build registry.** `scripts/build_registry.py` scans `data/rollouts/` and writes `data/rollout_registry.csv` — a flat index of all episodes with their metadata fields. Re-run this every time the Robotics team adds new episodes.

**Step 3 — Generate preference pairs.** Three independent pipelines each read the registry and produce a `(traj_A, traj_B, label)` CSV in `data/preferences/`:

| Source | Script | Output | Notes |
|---|---|---|---|
| **Scripted** | `gen_scripted_pairs.py` | `scripted_pairs.csv` | Free, instant |
| **Human** | `gen_human_pairs.py` | `human_pairs.csv` | ~1 min/pair |
| **VLM (Gemini) — default** | `gen_vlm_pairs.py` | `vlm_pairs.csv` | Requires `GOOGLE_API_KEY` |
| **VLM (local HF) — backup** | `gen_vlm_pairs.py --backend hf` | `vlm_pairs.csv` | Requires GPU, no API cost |

Label values: `1.0` = traj_A preferred, `0.0` = traj_B preferred, `0.5` = tie.

Scripted labels come from `metadata.json` fields (`success_label`, and optionally `collision` and `distance_to_goal`). Human and VLM labels come from watching the mp4 videos.

**Step 4 — Train reward models.** `src/reward_model/train.py` (to be built) trains one reward model per preference CSV, resulting in three independent reward functions.

**Step 5 — Compare.** `src/reward_model/evaluate.py` (to be built) measures:
- **Accuracy** — how well each reward model predicts held-out human labels
- **Agreement** — pairwise overlap between the three preference sources
- **Average reward score** — mean reward assigned to successful episodes

Results go to `results/results_table.csv` and figures to `results/figures/`.
