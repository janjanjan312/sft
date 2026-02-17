"""
Run LLaMA-Factory training and save a clean log for submission.

Usage:
  python scripts/run_llamafactory_train.py --config configs/qwen2_5_0_5b_sft_full.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/qwen2_5_0_5b_sft_full.yaml",
        help="Path to LLaMA-Factory YAML config.",
    )
    ap.add_argument(
        "--log",
        default="artifacts/training_logs/train.log",
        help="Where to write combined stdout/stderr log.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    cmd = ["llamafactory-cli", "train", args.config]

    header = (
        f"[run_llamafactory_train] start_time={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[run_llamafactory_train] cmd={' '.join(cmd)}\n"
        f"[run_llamafactory_train] note: set CUDA_VISIBLE_DEVICES if needed\n\n"
    )

    with open(args.log, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="")

        rc = proc.wait()
        f.write(f"\n[run_llamafactory_train] exit_code={rc}\n")
        f.flush()

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()

