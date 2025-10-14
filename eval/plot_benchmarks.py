#!/usr/bin/env python3
"""Visualize evaluation metrics produced by `eval/run_benchmarks.py`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from statistics import NormalDist

import matplotlib.pyplot as plt


def t_multiplier(n: int | None, confidence: float) -> float:
    """Return the 2-sided z critical value for the requested confidence."""
    if not confidence:
        return 0.0
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be between 0 and 1 (exclusive).")
    return NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def find_latest_run(runs_dir: Path) -> Path:
    candidates = sorted(
        runs_dir.glob("eval_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run files found in {runs_dir}")
    return candidates[0]


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clean_label(task_name: str, alias: str | None) -> str:
    label = alias.strip() if alias else task_name
    if label.startswith("- "):
        label = label[2:]
    label = label.replace("_", " ")
    return label


_METRIC_PRIORITY = (
    "acc",
    "accuracy",
    "exact_match",
    "f1",
    "rouge",
    "bleu",
)


def _infer_stderr_key(metric_key: str) -> str:
    if "," in metric_key:
        base, suffix = metric_key.split(",", 1)
        return f"{base}_stderr,{suffix}"
    return f"{metric_key}_stderr"


def resolve_metric_pair(
    metrics: Dict[str, Any],
    metric_key: str,
    stderr_key: str,
) -> Tuple[float, float]:
    mean = metrics.get(metric_key)
    stderr = metrics.get(stderr_key)
    if isinstance(mean, (int, float)):
        return float(mean), float(stderr or 0.0)

    numeric_entries: List[Tuple[int, str]] = []
    for key, value in metrics.items():
        if key == "alias" or "stderr" in key:
            continue
        if not isinstance(value, (int, float)):
            continue
        lowered = key.lower()
        priority = len(_METRIC_PRIORITY)
        for idx, prefix in enumerate(_METRIC_PRIORITY):
            if lowered.startswith(prefix):
                priority = idx
                break
        numeric_entries.append((priority, key))

    if not numeric_entries:
        raise KeyError(metric_key)

    _, chosen_key = min(numeric_entries, key=lambda item: (item[0], item[1]))
    mean_val = metrics[chosen_key]
    stderr_key_guess = _infer_stderr_key(chosen_key)
    stderr_val = metrics.get(stderr_key_guess, 0.0)
    return float(mean_val), float(stderr_val or 0.0)


def collect_metrics(
    results: Dict[str, Any],
    n_samples: Dict[str, Any],
    metric_key: str,
    stderr_key: str,
    focus_prefix: str | None = None,
    include_tasks: Iterable[str] | None = None,
    exclude_prefixes: Iterable[str] | None = None,
) -> List[Tuple[str, float, float, int | None]]:
    include_set = set(include_tasks or [])
    exclude_tuple = tuple(exclude_prefixes or ())
    collected: List[Tuple[str, float, float, int | None]] = []
    for task_name, metrics in results.items():
        if (
            focus_prefix
            and not task_name.startswith(focus_prefix)
            and task_name not in include_set
        ):
            continue
        if (
            task_name not in include_set
            and exclude_tuple
            and any(task_name.startswith(prefix) for prefix in exclude_tuple)
        ):
            continue
        if not isinstance(metrics, dict):
            continue
        try:
            mean, stderr = resolve_metric_pair(metrics, metric_key, stderr_key)
        except KeyError:
            continue
        alias = metrics.get("alias")
        label = clean_label(task_name, alias)
        n_info = n_samples.get(task_name, {})
        n_effective = n_info.get("effective")
        collected.append((label, float(mean), float(stderr or 0.0), n_effective))
    if not collected:
        raise ValueError("No metrics matched the provided filters.")
    return collected


def build_chart(
    metrics: Iterable[Tuple[str, float, float, int | None]],
    run_path: Path,
    output: Path,
    confidence: float,
) -> None:
    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []

    for label, mean, stderr, n_effective in metrics:
        labels.append(label)
        values.append(mean * 100.0)
        multiplier = t_multiplier(n_effective, confidence)
        errors.append(stderr * multiplier * 100.0)

    fig_height = max(4.0, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(labels)), fig_height))
    positions = range(len(labels))
    ax.bar(positions, values, yerr=errors, capsize=4.0, color="#4C72B0")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Evaluation Metrics ({run_path.name})")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize evaluation runs.")
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Path to a run JSON file. Defaults to the most recent `runs/eval_*.json`.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for the generated figure (PNG). Defaults to alongside the run JSON.",
    )
    parser.add_argument(
        "--metric-key",
        default="acc,none",
        help="Metric field to visualize (default: acc,none).",
    )
    parser.add_argument(
        "--stderr-key",
        default="acc_stderr,none",
        help="Standard error field for error bars (default: acc_stderr,none).",
    )
    parser.add_argument(
        "--focus-prefix",
        default=None,
        help="Only include tasks whose names start with the provided prefix (e.g. mmlu_).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for error bars (default: 0.95).",
    )
    parser.add_argument(
        "--include-task",
        action="append",
        default=None,
        help="Explicit task ids to include even if filtered out by prefix (e.g. mmlu for the official aggregate).",
    )
    parser.add_argument(
        "--show-mmlu-subtasks",
        action="store_true",
        help="Include individual MMLU subject results (defaults to showing only the aggregate).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit the number of tasks shown (keeps the highest-accuracy tasks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir
    run_path = args.run or find_latest_run(runs_dir)
    if not run_path.is_file():
        raise FileNotFoundError(f"Run file {run_path} does not exist.")

    data = load_results(run_path)
    results = data.get("results", {})
    n_samples = data.get("n-samples", {})
    focus_prefix = args.focus_prefix
    include_tasks = list(args.include_task or [])
    exclude_prefixes = () if args.show_mmlu_subtasks else ("mmlu_",)

    metrics = collect_metrics(
        results,
        n_samples,
        args.metric_key,
        args.stderr_key,
        focus_prefix=focus_prefix,
        include_tasks=include_tasks,
        exclude_prefixes=exclude_prefixes,
    )

    if args.max_tasks is not None and args.max_tasks > 0:
        metrics = sorted(metrics, key=lambda item: item[1], reverse=True)[: args.max_tasks]

    output_path = args.output or run_path.with_suffix(".png")
    build_chart(metrics, run_path, output_path, args.confidence)

    print(f"[viz] Saved chart with {len(metrics)} bars to {output_path}")


if __name__ == "__main__":
    main()
