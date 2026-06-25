"""Generate VLM preference pairs using the Google Gemini API."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preferences.vlm_gemini import generate_vlm_pairs

if __name__ == "__main__":
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate VLM preference pairs via Gemini.")
    parser.add_argument(
        "--registry", type=Path,
        default=project_root / "data" / "rollout_registry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "data" / "preferences" / "vlm_pairs.csv",
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API key (falls back to GOOGLE_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument(
        "--prompt-path", type=Path,
        default=project_root / "prompts" / "gemini_pref.txt",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in .env or pass --api-key")

    generate_vlm_pairs(
        args.registry, args.output,
        api_key, args.model, str(args.prompt_path), args.max_pairs,
    )
