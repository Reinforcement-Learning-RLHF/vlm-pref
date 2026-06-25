"""
VLM preference labeling using local HuggingFace models (Qwen3-VL, Qwen2-VL).

Importable interface:
    load_vlm(family, model_name, device, dtype) -> (model, processor)
    label_pair(video_path_1, video_path_2, model, processor, prompt_path) -> str
    generate_vlm_pairs(registry_csv, output_csv, ...) -> int
"""

import csv
from pathlib import Path

import torch
from transformers import AutoProcessor

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_vlm(
    family: str = "qwen3",
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str | None = None,
    dtype: str = "bfloat16",
):
    """Load and return (model, processor)."""
    if family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as VLM
    elif family == "qwen2":
        from transformers import Qwen2VLForConditionalGeneration as VLM
    else:
        raise ValueError(f"Unsupported model family: {family}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def _run_inference(model, processor, messages: list) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def label_pair(
    video_path_1: str,
    video_path_2: str,
    model,
    processor,
    prompt_path: str | None = None,
) -> str:
    """
    Run pairwise preference inference on two videos.
    Returns the raw model output string (contains 'Video 1' or 'Video 2').
    """
    if prompt_path is None:
        prompt_path = PROMPTS_DIR / "system_pref.txt"

    with open(prompt_path) as f:
        system_prompt = f.read()

    user_content = [
        {"type": "text", "text": "Video 1"},
        {"type": "video", "video": video_path_1},
        {"type": "text", "text": "Video 2"},
        {"type": "video", "video": video_path_2},
    ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    return _run_inference(model, processor, messages)


def _parse_winner(raw_output: str) -> float:
    """Parse 'Video 1 is better' style output → 1.0, 0.0, or 0.5."""
    lower = raw_output.lower()
    if "video 1" in lower and "video 2" not in lower:
        return 1.0
    elif "video 2" in lower and "video 1" not in lower:
        return 0.0
    elif "video 1" in lower and "video 2" in lower:
        # Both mentioned; look for explicit winner signal
        idx1 = lower.index("video 1")
        idx2 = lower.index("video 2")
        return 1.0 if idx1 < idx2 else 0.0
    return 0.5


def generate_vlm_pairs(
    registry_csv: str,
    output_csv: str,
    family: str = "qwen3",
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str | None = None,
    dtype: str = "bfloat16",
    prompt_path: str | None = None,
    max_pairs: int | None = None,
) -> int:
    """
    Load a local VLM, label pairwise preferences from registry_csv, write to output_csv.
    Returns number of pairs labeled.
    """
    registry_csv = Path(registry_csv)
    output_csv = Path(output_csv)

    model, processor = load_vlm(family, model_name, device, dtype)

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
            raw = label_pair(a["video_path"], b["video_path"], model, processor, prompt_path)
            label = _parse_winner(raw)
            writer.writerow({"traj_A": a["rollout_id"], "traj_B": b["rollout_id"], "label": label})

    print(f"Wrote {len(pairs)} labeled pairs to {output_csv}")
    return len(pairs)
