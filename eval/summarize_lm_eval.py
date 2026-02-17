"""
Summarize lm_eval json outputs to a single CSV for base vs SFT comparison.

Example:
  python eval/summarize_lm_eval.py \
    --base_dir artifacts/eval/base \
    --sft_dir artifacts/eval/sft \
    --out_csv artifacts/eval/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional, Tuple


TASKS_FEWSHOT = {
    "mmlu": 5,
    "arc_easy": 0,
    "arc_challenge": 25,
    "hellaswag": 10,
    "winogrande": 5,
    "truthfulqa_mc2": 0,
    "piqa": 0,
    "boolq": 0,
}


PREFERRED_METRICS = {
    "mmlu": ["acc,none", "acc"],
    "arc_easy": ["acc,none", "acc_norm,none", "acc", "acc_norm"],
    "arc_challenge": ["acc,none", "acc_norm,none", "acc", "acc_norm"],
    "hellaswag": ["acc_norm,none", "acc,none", "acc_norm", "acc"],
    "winogrande": ["acc,none", "acc"],
    "truthfulqa_mc2": ["mc2,none", "mc2"],
    "piqa": ["acc,none", "acc"],
    "boolq": ["acc,none", "acc"],
}


def load_task_score(path: str, task: str) -> Tuple[Optional[str], Optional[float]]:
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    results = data.get("results", {})
    task_res: Dict[str, Any] = results.get(task, {}) if isinstance(results, dict) else {}

    if not task_res:
        # some lm_eval versions may store flat results
        task_res = results if isinstance(results, dict) else {}

    prefs = PREFERRED_METRICS.get(task, [])
    for k in prefs:
        if k in task_res and isinstance(task_res[k], (int, float)):
            return k, float(task_res[k])

    # fallback: pick first numeric metric
    for k, v in task_res.items():
        if isinstance(v, (int, float)):
            return str(k), float(v)

    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--sft_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    rows = []
    for task, fewshot in TASKS_FEWSHOT.items():
        fname = f"{task}_{fewshot}shot.json"
        base_path = os.path.join(args.base_dir, fname)
        sft_path = os.path.join(args.sft_dir, fname)
        base_metric, base_score = load_task_score(base_path, task)
        sft_metric, sft_score = load_task_score(sft_path, task)
        metric = sft_metric or base_metric
        delta = None
        if base_score is not None and sft_score is not None:
            delta = sft_score - base_score

        rows.append(
            {
                "task": task,
                "fewshot": fewshot,
                "metric": metric,
                "base_score": base_score,
                "sft_score": sft_score,
                "delta": delta,
            }
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["task", "fewshot", "metric", "base_score", "sft_score", "delta"],
        )
        w.writeheader()
        w.writerows(rows)

    # also write a markdown table (handy for report)
    md_path = os.path.splitext(args.out_csv)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| task | fewshot | metric | base | sft | delta |\n")
        f.write("|---|---:|---|---:|---:|---:|\n")
        for r in rows:
            base_str = "" if r["base_score"] is None else f"{r['base_score']:.4f}"
            sft_str = "" if r["sft_score"] is None else f"{r['sft_score']:.4f}"
            delta_str = "" if r["delta"] is None else f"{r['delta']:.4f}"
            metric_str = r["metric"] or ""
            f.write(
                f"| {r['task']} | {r['fewshot']} | {metric_str} | {base_str} | {sft_str} | {delta_str} |\n"
            )

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()

