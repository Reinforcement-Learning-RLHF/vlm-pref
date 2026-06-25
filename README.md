# Preference Feedback for Robot Manipulation

Compare three sources of preference feedback — scripted rules, human labels, and VLM scoring — to train and evaluate reward models for robot pick-and-place tasks on a real SO-101 arm.

## Team Split

| Team | Responsibility |
|---|---|
| **Robotics** | Record episodes with the SO-101 arm via LeRobot; deliver mp4 + `metadata.json` per episode into `data/rollouts/` |
| **ML (this repo)** | Preference labeling, reward model training, and comparison |

## Pipeline

```
data/rollouts/                    ← episodes from Robotics team
        ↓  scripts/build_registry.py
data/rollout_registry.csv
        ↓  scripts/gen_*_pairs.py  (3 independent sources)
data/preferences/
    scripted_pairs.csv            ← rule-based from metadata.json
    human_pairs.csv               ← interactive human labeling
    vlm_pairs.csv                 ← vision-language model scoring
        ↓  src/reward_model/train.py  (TODO)
one reward model per source
        ↓  src/reward_model/evaluate.py  (TODO)
results/results_table.csv + results/figures/
```

See [docs/pipeline.md](docs/pipeline.md) for a full step-by-step description.

## Setup

```bash
python -m venv ~/robenv
source ~/robenv/bin/activate      # Windows: ~/robenv/Scripts/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY
```

## Running the Pipeline

**Step 1 — Build the rollout registry** (re-run after every new episode batch):

```bash
python scripts/build_registry.py
```

**Step 2 — Generate preference pairs** (run one or all three):

```bash
# Scripted — instant, no API or GPU needed
python scripts/gen_scripted_pairs.py

# Human — interactive terminal session (opens videos on request)
python scripts/gen_human_pairs.py --sample-size 50

# VLM — Gemini (recommended, requires GOOGLE_API_KEY in .env)
python scripts/gen_vlm_pairs.py

# VLM — local HuggingFace model (backup, requires GPU, no API cost)
python scripts/gen_vlm_pairs.py --backend hf --model Qwen/Qwen3-VL-2B-Instruct
```

All scripts write to `data/preferences/` and accept `--help` for full options.

**Step 3 — Train reward models** *(not yet implemented)*:

```bash
python src/reward_model/train.py --pairs data/preferences/scripted_pairs.csv
```

**Step 4 — Compare results** *(not yet implemented)*:

```bash
python src/reward_model/evaluate.py
```

## Utility Scripts

```bash
# Convert HDF5 image arrays to mp4 (for pre-LeRobot data)
python scripts/convert_hdf5_to_mp4.py <hdf5_path> --output output.mp4

# Extract keyframes from a video using optical flow
python scripts/extract_keyframes.py --video_path path/to/video.mp4
```

## Project Structure

```
data/rollouts/          Raw episodes (mp4 + metadata.json per subfolder)
data/preferences/       Generated preference CSVs (one per labeling source)
prompts/                VLM system prompts (Gemini and HuggingFace variants)
src/data/               Registry building logic
src/preferences/        Preference labeling modules (scripted, human, vlm)
src/reward_model/       Reward model training and evaluation (TODO)
scripts/                Runnable CLI entry points
results/                Reward curves and final comparison table
docs/                   Format specs and pipeline documentation
archive/                Superseded scripts and experimental code
```

## Data Format

See [docs/rollout_format.md](docs/rollout_format.md) for the full episode format spec, required `metadata.json` fields, and registry CSV schema.
