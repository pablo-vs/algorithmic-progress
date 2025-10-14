# eval/run_benchmarks.py
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
from huggingface_hub import login
from lm_eval import evaluator
from lm_eval.utils import handle_non_serializable

# Load .env if present (so HF_TOKEN is available)
load_dotenv()

# Optional: authenticate for gated models (Llama, Gemma, etc.)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    # Avoid writing into git credentials; keep it local to HF cache
    login(token=hf_token, add_to_git_credential=False)
else:
    print("[warn] HF_TOKEN not set — gated models may fail to download.")

DEFAULT_TASKS = ["mmlu", "truthfulqa_mc2", "arc_easy", "gsm8k"]


def run_eval(
    model_name: str,
    tasks: List[str],
    limit: int | None = None,
    seed: int = 123,
    batch_size: int | None = None,
    device: str | None = None,
    extra_model_args: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Quick evaluation wrapper over lm-eval for decoder-only (causal) transformers.
    Supports subsampling via `limit` and deterministic selection via `seed`.
    """
    model_args_items = [f"pretrained={model_name}"]
    if batch_size is not None:
        model_args_items.append(f"batch_size={batch_size}")
    if device is not None:
        model_args_items.append(f"device={device}")
    if extra_model_args:
        for k, v in extra_model_args.items():
            model_args_items.append(f"{k}={v}")
    model_args_str = ",".join(model_args_items)

    results = evaluator.simple_evaluate(
        model="hf",                  # hf loader handles causal decoder-only models
        model_args=model_args_str,
        tasks=tasks,
        limit=limit,                 # e.g., 20–30 for quick checks
        random_seed=seed,            # controls the sampled subset
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=handle_non_serializable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick LLM benchmarks via lm-eval (causal-only).")
    parser.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Llama-3-8b-instruct")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                        help="Comma-separated tasks, e.g. mmlu,truthfulqa_mc,arc_easy,gsm8k")
    parser.add_argument("--limit", type=int, default=25, help="Max examples per task.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for subsampling.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help='e.g., "cuda:0" or "cpu"')
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = run_eval(
        model_name=args.model,
        tasks=tasks,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = args.output or f"runs/eval_{timestamp}.json"
    save_results(results, out_path)
    print(f"[eval] Saved results to {out_path}")


if __name__ == "__main__":
    main()
