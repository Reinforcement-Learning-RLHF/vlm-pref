"""
Convenience wrapper for VLM preference labeling.

Defaults to Gemini (recommended). Use --backend hf for local HuggingFace inference.

Usage:
    python scripts/gen_vlm_pairs.py                              # Gemini (default)
    python scripts/gen_vlm_pairs.py --backend hf                 # local HF model
    python scripts/gen_vlm_pairs.py --backend gemini --max-pairs 20
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_gemini(args):
    from src.preferences.vlm_gemini import generate_vlm_pairs

    load_dotenv()
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in .env or pass --api-key")

    generate_vlm_pairs(
        registry_csv=args.registry,
        output_csv=args.output,
        api_key=api_key,
        model=args.model or "gemini-2.5-flash",
        prompt_path=str(args.prompt_path) if args.prompt_path else None,
        max_pairs=args.max_pairs,
    )


def run_hf(args):
    from src.preferences.vlm_hf import generate_vlm_pairs

    generate_vlm_pairs(
        registry_csv=args.registry,
        output_csv=args.output,
        family=args.family,
        model_name=args.model or "Qwen/Qwen3-VL-2B-Instruct",
        device=args.device,
        dtype=args.dtype,
        prompt_path=str(args.prompt_path) if args.prompt_path else None,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VLM preference pairs (Gemini by default)."
    )
    parser.add_argument(
        "--backend", choices=["gemini", "hf"], default="gemini",
        help="Which VLM backend to use: gemini (default) or hf (local HuggingFace)",
    )
    parser.add_argument(
        "--registry", type=Path,
        default=PROJECT_ROOT / "data" / "rollout_registry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "preferences" / "vlm_pairs.csv",
    )
    parser.add_argument(
        "--prompt-path", type=Path, default=None,
        help="Path to prompt file (uses backend default if not set)",
    )
    parser.add_argument("--max-pairs", type=int, default=None)

    # Gemini-specific
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (falls back to GOOGLE_API_KEY env var)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (gemini-2.5-flash or Qwen/Qwen3-VL-2B-Instruct)")

    # HF-specific
    parser.add_argument("--family", type=str, default="qwen3",
                        help="HF model family: qwen3 or qwen2")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu or cuda (HF only, auto-detected if not set)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype (HF only)")

    args = parser.parse_args()

    if args.backend == "gemini":
        run_gemini(args)
    else:
        run_hf(args)
