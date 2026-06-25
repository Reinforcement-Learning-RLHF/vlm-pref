"""
Unified pair evaluator that routes to the right preference source.

Importable interface:
    evaluate_pair(rollout_a, rollout_b, evaluator, **kwargs) -> dict
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLLOUTS_DIR = PROJECT_ROOT / "data" / "rollouts"


def _load_metadata(rollout_id: str) -> dict:
    meta_path = ROLLOUTS_DIR / rollout_id / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json found for rollout '{rollout_id}' at {meta_path}"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_video(metadata: dict) -> str:
    """Return the absolute path to a rollout's agentview video."""
    rel = metadata.get("video_path", "")
    abs_path = PROJECT_ROOT / rel
    if not abs_path.exists():
        raise FileNotFoundError(f"Video not found: {abs_path}")
    return str(abs_path)


def evaluate_pair(
    rollout_a: str,
    rollout_b: str,
    evaluator: str = "scripted",
    **kwargs: Any,
) -> dict:
    """
    Evaluate a pair of rollouts using the specified evaluator.

    Args:
        rollout_a:  rollout_id of the first episode (folder name under data/rollouts/)
        rollout_b:  rollout_id of the second episode
        evaluator:  "scripted", "vlm_gemini", or "human"
        **kwargs:   evaluator-specific options:
                      vlm_gemini -> api_key (required), model, prompt_path
                      scripted   -> (none)
                      human      -> (none)

    Returns:
        {
            "traj_A":         rollout_a,
            "traj_B":         rollout_b,
            "label":          float  (1.0=A better, 0.0=B better, 0.5=tie),
            "evaluator_type": str,
            "timestamp":      str (ISO 8601 UTC),
            ...evaluator-specific extra fields (e.g. comments from Gemini)
        }
    """
    meta_a = _load_metadata(rollout_a)
    meta_b = _load_metadata(rollout_b)

    if evaluator == "scripted":
        from src.preferences.scripted import _score_rollout

        score_a = _score_rollout(meta_a)
        score_b = _score_rollout(meta_b)
        if score_a > score_b:
            label = 1.0
        elif score_b > score_a:
            label = 0.0
        else:
            label = 0.5
        extra = {}

    elif evaluator == "vlm_gemini":
        from src.preferences.vlm_gemini import label_pair

        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError(
                "vlm_gemini evaluator requires api_key. "
                "Pass api_key=... or set GOOGLE_API_KEY in .env"
            )
        result = label_pair(
            video_path_1=_resolve_video(meta_a),
            video_path_2=_resolve_video(meta_b),
            api_key=api_key,
            model=kwargs.get("model", "gemini-2.5-flash"),
            prompt_path=kwargs.get("prompt_path"),
        )
        label = result["score"]
        extra = {k: v for k, v in result.items() if k != "score"}

    elif evaluator == "human":
        from src.preferences.human import label_pair

        raw = label_pair(_resolve_video(meta_a), _resolve_video(meta_b))
        label = 0.5 if raw is None else raw  # None means skipped -> tie
        extra = {}

    else:
        raise ValueError(
            f"Unknown evaluator '{evaluator}'. "
            "Valid options: 'scripted', 'vlm_gemini', 'human'."
        )

    return {
        "traj_A": rollout_a,
        "traj_B": rollout_b,
        "label": label,
        "evaluator_type": evaluator,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
