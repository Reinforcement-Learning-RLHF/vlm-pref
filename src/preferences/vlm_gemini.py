"""
VLM preference labeling using Google Gemini API.

Importable interface:
    label_pair(video_path_1, video_path_2, api_key, model, prompt, prompt_path) -> dict
    generate_vlm_pairs(registry_csv, output_csv, api_key, model, prompt_path, max_pairs) -> int
"""

import csv
import time
from pathlib import Path

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PreferenceLabel(BaseModel):
    score: float = Field(
        description="Score for Video 1 relative to Video 2: 0 if Video 1 is worse, 0.5 if equally good, 1 if Video 1 is better"
    )
    comment: str = Field(description="Brief qualitative explanation of the preference decision")


def _wait_for_active(client, video_file):
    while video_file.state.name != "ACTIVE":
        print(f"  Waiting for {video_file.name} to become ACTIVE...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    return video_file


def label_pair(
    video_path_1: str,
    video_path_2: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str | None = None,
    prompt_path: str | None = None,
) -> dict:
    """
    Label one (A, B) video pair using Gemini. Runs both orderings to reduce position bias.

    Returns dict: score, score_ab, score_ba_flipped, comment_ab, comment_ba.
    score=1 → A is better; score=0 → B is better; score=0.5 → tie.
    """
    if prompt is None:
        path = Path(prompt_path) if prompt_path else PROMPTS_DIR / "gemini_pref.txt"
        with open(path) as f:
            prompt = f.read()

    upload_client = genai.Client(api_key=api_key)
    print(f"  Uploading {video_path_1} ...")
    vf1 = upload_client.files.upload(file=video_path_1)
    print(f"  Uploading {video_path_2} ...")
    vf2 = upload_client.files.upload(file=video_path_2)

    vf1 = _wait_for_active(upload_client, vf1)
    vf2 = _wait_for_active(upload_client, vf2)

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    structured_llm = llm.with_structured_output(PreferenceLabel)

    def _query(first, second):
        message = HumanMessage(content=[
            {"type": "text", "text": "Video 1:"},
            {"type": "media", "mime_type": "video/mp4", "file_uri": first.uri},
            {"type": "text", "text": "Video 2:"},
            {"type": "media", "mime_type": "video/mp4", "file_uri": second.uri},
            {"type": "text", "text": prompt},
        ])
        return structured_llm.invoke([message])

    result_ab = _query(vf1, vf2)
    result_ba = _query(vf2, vf1)
    combined_score = (result_ab.score + (1 - result_ba.score)) / 2

    return {
        "score": combined_score,
        "score_ab": result_ab.score,
        "score_ba_flipped": 1 - result_ba.score,
        "comment_ab": result_ab.comment,
        "comment_ba": result_ba.comment,
    }


def generate_vlm_pairs(
    registry_csv: str,
    output_csv: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt_path: str | None = None,
    max_pairs: int | None = None,
) -> int:
    """
    Read rollout registry, generate pairwise Gemini preference labels, write to output_csv.

    Pairs successes against failures. Returns number of pairs labeled.
    """
    registry_csv = Path(registry_csv)
    output_csv = Path(output_csv)

    if prompt_path is None:
        prompt_path = str(PROMPTS_DIR / "gemini_pref.txt")

    with open(prompt_path) as f:
        prompt = f.read()

    with open(registry_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    successes = [r for r in rows if r.get("success_label", "").lower() == "true"]
    failures = [r for r in rows if r.get("success_label", "").lower() == "false"]
    pairs = [(s, fail) for s in successes for fail in failures]

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["traj_A", "traj_B", "label"])
        writer.writeheader()

        for i, (a, b) in enumerate(pairs):
            print(f"Labeling pair {i + 1}/{len(pairs)}: {a['rollout_id']} vs {b['rollout_id']}")
            result = label_pair(a["video_path"], b["video_path"], api_key, model, prompt=prompt)
            writer.writerow({
                "traj_A": a["rollout_id"],
                "traj_B": b["rollout_id"],
                "label": result["score"],
            })

    print(f"Wrote {len(pairs)} labeled pairs to {output_csv}")
    return len(pairs)
