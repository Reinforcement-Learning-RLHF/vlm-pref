# TODO (John): Entry point for the Streamlit human labeling app.
#
# Run with:
#   streamlit run app/main.py
#
# Suggested flow:
#   1. Load data/rollout_registry.csv to get available episodes
#   2. Pick a random success/failure pair not yet labeled in data/preferences/human_pairs.csv
#   3. Show the two agentview videos side by side
#   4. User clicks "A is better", "B is better", or "Tie"
#   5. Append result to data/preferences/human_pairs.csv (traj_A, traj_B, label)
#   6. Show progress (X of Y pairs labeled)
#
# The labeling logic is already implemented in src/preferences/human.py
# (terminal version). This app is the video-capable replacement.

import streamlit as st

st.set_page_config(page_title="Preference Labeler", layout="wide")
st.title("Preference Feedback Labeler")
st.info("App coming soon. See app/README.md for implementation notes.")
