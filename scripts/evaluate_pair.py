"""Evaluate a single rollout pair from the command line."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_demo import evaluate_pair

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate a pair of rollouts and print the result as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/evaluate_pair.py good_clean_slow bad_shaky
  python scripts/evaluate_pair.py good_clean_slow bad_shaky --evaluator vlm_gemini
  python scripts/evaluate_pair.py good_clean_slow bad_shaky --evaluator human
        """,
    )
    parser.add_argument("rollout_a", help="rollout_id of the first episode")
    parser.add_argument("rollout_b", help="rollout_id of the second episode")
    parser.add_argument(
        "--evaluator",
        choices=["scripted", "vlm_gemini", "human"],
        default="scripted",
        help="Which evaluator to use (default: scripted)",
    )
    parser.add_argument("--api-key", default=None, help="Gemini API key (vlm_gemini only)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model (vlm_gemini only)")
    parser.add_argument("--prompt-path", default=None, help="Path to prompt file (vlm_gemini only)")
    args = parser.parse_args()

    kwargs = {}
    if args.evaluator == "vlm_gemini":
        kwargs["api_key"] = args.api_key or os.getenv("GOOGLE_API_KEY")
        kwargs["model"] = args.model
        if args.prompt_path:
            kwargs["prompt_path"] = args.prompt_path

    result = evaluate_pair(args.rollout_a, args.rollout_b, args.evaluator, **kwargs)
    print(json.dumps(result, indent=2))
