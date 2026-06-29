# App (Human Labeling UI)

Owner: John

## What's here

A Streamlit app for collecting human preference labels by watching two robot videos and picking the better one. Replaces the terminal-based `scripts/gen_human_pairs.py` with a proper video UI.

## Structure

```
app/
    main.py          <- labeling interface (entry point)
    pages/
        results.py   <- results dashboard (label progress, agreement stats)
```

## Running

```bash
streamlit run app/main.py
```

## Key files to read first

- `src/preferences/human.py` -- human labeling logic implemented (terminal version); adapt for Streamlit
- `data/rollout_registry.csv` -- list of episodes with video paths and success labels
- `data/preferences/human_pairs.csv` -- output file; append (traj_A, traj_B, label) rows here

## Label format

| Column | Value |
|---|---|
| traj_A | rollout_id of the first video shown |
| traj_B | rollout_id of the second video shown |
| label | 1.0 = A is better, 0.0 = B is better, 0.5 = tie |
