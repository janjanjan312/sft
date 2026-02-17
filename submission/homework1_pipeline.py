"""
Homework1 end-to-end pipeline (data -> SFT -> eval).

This is provided as a single "complete script" for submission convenience.
You can also run individual scripts in `scripts/` and `eval/`.

Typical usage (on a CUDA machine):
  python submission/homework1_pipeline.py prepare_data --max_samples 250000
  python submission/homework1_pipeline.py train
  python submission/homework1_pipeline.py eval_base
  python submission/homework1_pipeline.py eval_sft --sft_model_dir artifacts/saves/qwen2.5-0.5b/full/sft
  python submission/homework1_pipeline.py summarize
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List


def _run(cmd: List[str]) -> None:
    print(f"[pipeline] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    _run(
        [
            "python",
            "scripts/prepare_openorca_sft.py",
            "--max_samples",
            str(args.max_samples),
            "--seed",
            str(args.seed),
        ]
    )


def cmd_train(_: argparse.Namespace) -> None:
    _run(["python", "scripts/run_llamafactory_train.py", "--config", "configs/qwen2_5_0_5b_sft_full.yaml"])


def cmd_eval_base(args: argparse.Namespace) -> None:
    _run(
        [
            "python",
            "eval/run_lm_eval.py",
            "--model_pretrained",
            "Qwen/Qwen2.5-0.5B",
            "--out_dir",
            args.out_dir,
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
        ]
    )


def cmd_eval_sft(args: argparse.Namespace) -> None:
    sft_dir = args.sft_model_dir
    if not os.path.exists(sft_dir):
        raise SystemExit(f"SFT model dir not found: {sft_dir}")
    _run(
        [
            "python",
            "eval/run_lm_eval.py",
            "--model_pretrained",
            sft_dir,
            "--out_dir",
            args.out_dir,
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
        ]
    )


def cmd_summarize(args: argparse.Namespace) -> None:
    _run(
        [
            "python",
            "eval/summarize_lm_eval.py",
            "--base_dir",
            args.base_dir,
            "--sft_dir",
            args.sft_dir,
            "--out_csv",
            args.out_csv,
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("prepare_data")
    p1.add_argument("--max_samples", type=int, default=250_000)
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=cmd_prepare_data)

    p2 = sub.add_parser("train")
    p2.set_defaults(func=cmd_train)

    p3 = sub.add_parser("eval_base")
    p3.add_argument("--out_dir", default="artifacts/eval/base")
    p3.add_argument("--device", default="cuda:0")
    p3.add_argument("--batch_size", type=int, default=4)
    p3.add_argument("--dtype", default="float16")
    p3.set_defaults(func=cmd_eval_base)

    p4 = sub.add_parser("eval_sft")
    p4.add_argument("--sft_model_dir", default="artifacts/saves/qwen2.5-0.5b/full/sft")
    p4.add_argument("--out_dir", default="artifacts/eval/sft")
    p4.add_argument("--device", default="cuda:0")
    p4.add_argument("--batch_size", type=int, default=4)
    p4.add_argument("--dtype", default="float16")
    p4.set_defaults(func=cmd_eval_sft)

    p5 = sub.add_parser("summarize")
    p5.add_argument("--base_dir", default="artifacts/eval/base")
    p5.add_argument("--sft_dir", default="artifacts/eval/sft")
    p5.add_argument("--out_csv", default="artifacts/eval/summary.csv")
    p5.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

