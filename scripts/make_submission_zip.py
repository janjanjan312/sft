"""
Create the submission zip (<10MB) without including model weights.

It stages a curated set of files into submission/_staging and zips them.

Usage:
  python scripts/make_submission_zip.py --student_id 12345678 --name ZhangSan
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree_if_exists(src_dir: str, dst_dir: str, *, suffix_allowlist: tuple[str, ...]) -> int:
    if not os.path.isdir(src_dir):
        return 0
    n = 0
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if suffix_allowlist and not fn.endswith(suffix_allowlist):
                continue
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, src_dir)
            dst = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n


def _zip_dir(src_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, src_dir)
                zf.write(full, rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", required=True)
    ap.add_argument("--name", required=True, help="Use English letters if possible (for filename).")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--out_dir", default="submission")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    out_dir = os.path.join(root, args.out_dir)
    staging = os.path.join(out_dir, "_staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    # 1) Code (scripts/config/eval)
    _copy_if_exists(os.path.join(root, "README.md"), os.path.join(staging, "README.md"))
    _copy_tree_if_exists(os.path.join(root, "scripts"), os.path.join(staging, "scripts"), suffix_allowlist=(".py",))
    _copy_tree_if_exists(os.path.join(root, "configs"), os.path.join(staging, "configs"), suffix_allowlist=(".yaml", ".yml"))
    _copy_tree_if_exists(os.path.join(root, "eval"), os.path.join(staging, "eval"), suffix_allowlist=(".py",))

    # 2) Logs / stats (small)
    _copy_if_exists(
        os.path.join(root, "artifacts", "training_logs", "train.log"),
        os.path.join(staging, "training_logs", "train.log"),
    )
    _copy_if_exists(
        os.path.join(root, "artifacts", "training_logs", "data_cleaning_stats.json"),
        os.path.join(staging, "training_logs", "data_cleaning_stats.json"),
    )
    _copy_if_exists(
        os.path.join(root, "artifacts", "saves", "qwen2.5-0.5b", "full", "sft", "plot_loss.png"),
        os.path.join(staging, "training_logs", "plot_loss.png"),
    )

    # 3) Eval results (json/csv/md)
    _copy_tree_if_exists(
        os.path.join(root, "artifacts", "eval"),
        os.path.join(staging, "eval_results"),
        suffix_allowlist=(".json", ".csv", ".md", ".log"),
    )

    # 4) Report
    _copy_if_exists(os.path.join(root, "report", "report.md"), os.path.join(staging, "report.md"))
    _copy_if_exists(os.path.join(root, "report", "report.pdf"), os.path.join(staging, "report.pdf"))

    # 5) Tiny demo dataset files (optional, keeps zip small)
    _copy_if_exists(
        os.path.join(root, "submission", "demo_openorca_sft_200.jsonl"),
        os.path.join(staging, "demo_openorca_sft_200.jsonl"),
    )
    _copy_if_exists(
        os.path.join(root, "submission", "dataset_info_demo.json"),
        os.path.join(staging, "dataset_info_demo.json"),
    )

    zip_name = f"{args.student_id}_{args.name}_AIMS5740_Homework1.zip"
    zip_path = os.path.join(out_dir, zip_name)
    _zip_dir(staging, zip_path)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Wrote: {zip_path} ({size_mb:.2f} MB)")
    if size_mb > 10:
        print("WARNING: zip > 10MB, please remove large files (do NOT include model weights).")


if __name__ == "__main__":
    main()

