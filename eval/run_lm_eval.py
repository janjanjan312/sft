"""
Run lm_eval for the 8 required tasks with fixed few-shot settings.

This script follows the tutorial/homework requirement:
- use lm_eval (recommended)
- set --apply_chat_template
- fixed num_fewshot per task

Example:
  python eval/run_lm_eval.py --model_pretrained "Qwen/Qwen2.5-0.5B" --out_dir artifacts/eval/base
  python eval/run_lm_eval.py --model_pretrained "artifacts/saves/qwen2.5-0.5b/full/sft" --out_dir artifacts/eval/sft
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Dict, List


TASKS_FEWSHOT: Dict[str, int] = {
    "mmlu": 5,
    "arc_easy": 0,
    "arc_challenge": 25,
    "hellaswag": 10,
    "winogrande": 5,
    "truthfulqa_mc2": 0,
    "piqa": 0,
    "boolq": 0,
}


def run_one(
    *,
    model_pretrained: str,
    task: str,
    num_fewshot: int,
    out_path: str,
    device: str,
    batch_size: int,
    dtype: str,
    extra_model_args: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    model_args = f"pretrained={model_pretrained},trust_remote_code=True,dtype={dtype}"
    if extra_model_args:
        # allow user to append ",key=value"
        model_args = model_args + ("," + extra_model_args.lstrip(","))

    cmd: List[str] = [
        "python",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--num_fewshot",
        str(num_fewshot),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--apply_chat_template",
        "--output_path",
        out_path,
    ]

    print(f"[lm_eval] task={task} fewshot={num_fewshot} out={out_path}")
    print(f"[lm_eval] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pretrained", required=True, help="HF model id or local path.")
    ap.add_argument("--out_dir", required=True, help="Directory to write task json results.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--extra_model_args",
        default="",
        help='Extra hf model_args appended to pretrained=..., e.g. ",max_length=2048".',
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "lm_eval_run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            f"model_pretrained={args.model_pretrained}\n"
            f"device={args.device}\n"
            f"batch_size={args.batch_size}\n"
            f"dtype={args.dtype}\n"
            f"extra_model_args={args.extra_model_args}\n"
        )

    for task, fewshot in TASKS_FEWSHOT.items():
        out_path = os.path.join(args.out_dir, f"{task}_{fewshot}shot.json")
        run_one(
            model_pretrained=args.model_pretrained,
            task=task,
            num_fewshot=fewshot,
            out_path=out_path,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            extra_model_args=args.extra_model_args,
        )


if __name__ == "__main__":
    main()

