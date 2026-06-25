"""Generate VLM preference pairs using a local HuggingFace model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preferences.vlm_hf import generate_vlm_pairs

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate VLM preference pairs via HuggingFace model.")
    parser.add_argument(
        "--registry", type=Path,
        default=project_root / "data" / "rollout_registry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "data" / "preferences" / "vlm_pairs.csv",
    )
    parser.add_argument("--family", type=str, default="qwen3",
                        help="Model family: qwen3 or qwen2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu or cuda (auto-detected if not set)")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--prompt-path", type=Path,
        default=project_root / "prompts" / "system_pref.txt",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    generate_vlm_pairs(
        args.registry, args.output,
        args.family, args.model, args.device, args.dtype,
        str(args.prompt_path), args.max_pairs,
    )
