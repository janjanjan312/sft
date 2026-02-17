"""
Prepare a deterministic OpenOrca subset for SFT (LLaMA-Factory Alpaca format).

Homework requirements recap:
- Model: Qwen/Qwen2.5-0.5B
- Dataset: Open-Orca/OpenOrca
- Deterministic 1M subset:
    revision = "e9c87b4"  # 2025-02-19
    split    = "train[:1_000_000]"
- Suggested final SFT size for 0.5B: 200k ~ 350k samples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import itertools
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple


OPENORCA_REVISION = "e9c87b4"  # 2025-02-19 (pinned for reproducibility)
OPENORCA_SUBSET_ROWS = 1_000_000


_WS_RE = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS_RE.sub(" ", s.strip())


def _has_long_char_run(s: str, run: int) -> bool:
    if run <= 1:
        return False
    current = 1
    last = ""
    for ch in s:
        if ch == last:
            current += 1
            if current >= run:
                return True
        else:
            last = ch
            current = 1
    return False


def _simple_repetition_ratio(text: str) -> float:
    """
    A cheap heuristic to filter templated spam:
    - tokenize by whitespace
    - compute max token frequency / total tokens
    """
    toks = [t for t in _norm_text(text).split(" ") if t]
    if not toks:
        return 1.0
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    return max(freq.values()) / max(1, len(toks))


@dataclass
class CleaningConfig:
    min_question_chars: int = 10
    min_response_chars: int = 20
    max_question_chars: int = 8_000
    max_response_chars: int = 16_000
    drop_if_char_run_ge: int = 120
    drop_if_rep_ratio_ge: float = 0.4
    rep_ratio_min_tokens: int = 80


@dataclass
class CleaningStats:
    source_rows_requested: int
    source_revision: str
    seen_rows: int = 0
    kept_rows: int = 0
    dropped_empty: int = 0
    dropped_length: int = 0
    dropped_char_run: int = 0
    dropped_repetition: int = 0
    dropped_dedup: int = 0
    dropped_other: int = 0


def _is_valid_sample(
    question: str, response: str, cfg: CleaningConfig, stats: CleaningStats
) -> bool:
    if not question or not response:
        stats.dropped_empty += 1
        return False

    q = _norm_text(question)
    r = _norm_text(response)
    if not q or not r:
        stats.dropped_empty += 1
        return False

    if (
        len(q) < cfg.min_question_chars
        or len(r) < cfg.min_response_chars
        or len(q) > cfg.max_question_chars
        or len(r) > cfg.max_response_chars
    ):
        stats.dropped_length += 1
        return False

    if _has_long_char_run(q, cfg.drop_if_char_run_ge) or _has_long_char_run(
        r, cfg.drop_if_char_run_ge
    ):
        stats.dropped_char_run += 1
        return False

    # Only apply repetition heuristic to long-ish answers
    r_toks = [t for t in _norm_text(r).split(" ") if t]
    if len(r_toks) >= cfg.rep_ratio_min_tokens:
        if _simple_repetition_ratio(r) >= cfg.drop_if_rep_ratio_ge:
            stats.dropped_repetition += 1
            return False

    return True


def _dedup_key(question: str, response: str) -> bytes:
    # Deterministic dedup based on normalized (question, response)
    payload = (_norm_text(question).lower() + "\n" + _norm_text(response).lower()).encode(
        "utf-8", errors="ignore"
    )
    return hashlib.sha1(payload).digest()


def iter_openorca_1m_subset() -> Iterable[Dict[str, Any]]:
    """
    Load the deterministic 1M subset as specified by the homework.
    """
    from datasets import load_dataset

    # Use streaming to avoid materializing 1M rows in RAM/disk (Colab-friendly).
    # Deterministic because we pin revision and take the first N rows in order.
    ds = load_dataset(
        "Open-Orca/OpenOrca",
        split="train",
        revision=OPENORCA_REVISION,
        streaming=True,
    )
    for row in itertools.islice(ds, OPENORCA_SUBSET_ROWS):
        yield row


def to_alpaca_jsonl_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenOrca fields (HF dataset card):
    - question
    - response
    - system_prompt
    - id
    """
    return {
        "instruction": _norm_text(str(row.get("question", ""))),
        "input": "",
        "output": _norm_text(str(row.get("response", ""))),
        "system": _norm_text(str(row.get("system_prompt", ""))),
        "history": [],
        "source_id": str(row.get("id", "")),
    }


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="artifacts/llamafactory_data/openorca_sft_clean.jsonl",
        help="Output jsonl path (Alpaca format).",
    )
    ap.add_argument(
        "--dataset_name",
        default="openorca_sft_clean",
        help="Dataset name used by LLaMA-Factory (dataset_info.json key).",
    )
    ap.add_argument(
        "--dataset_dir",
        default="artifacts/llamafactory_data",
        help="Directory holding dataset file + dataset_info.json for LLaMA-Factory.",
    )
    ap.add_argument(
        "--stats",
        default="artifacts/training_logs/data_cleaning_stats.json",
        help="Where to write cleaning stats JSON.",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=250_000,
        help="Final number of samples after filtering (suggested 200k~350k).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--progress_every",
        type=int,
        default=20_000,
        help="Print progress every N rows while scanning (0 to disable).",
    )

    # Cleaning thresholds
    ap.add_argument("--min_question_chars", type=int, default=10)
    ap.add_argument("--min_response_chars", type=int, default=20)
    ap.add_argument("--max_question_chars", type=int, default=8_000)
    ap.add_argument("--max_response_chars", type=int, default=16_000)
    ap.add_argument("--drop_if_char_run_ge", type=int, default=120)
    ap.add_argument("--drop_if_rep_ratio_ge", type=float, default=0.4)
    ap.add_argument("--rep_ratio_min_tokens", type=int, default=80)

    ap.add_argument(
        "--demo_out",
        default="submission/demo_openorca_sft_200.jsonl",
        help="Write a tiny demo dataset (200 rows) for submission/reference.",
    )
    ap.add_argument(
        "--demo_dataset_info_out",
        default="submission/dataset_info_demo.json",
        help="Write a demo dataset_info.json entry (for the demo jsonl).",
    )

    args = ap.parse_args()

    cfg = CleaningConfig(
        min_question_chars=args.min_question_chars,
        min_response_chars=args.min_response_chars,
        max_question_chars=args.max_question_chars,
        max_response_chars=args.max_response_chars,
        drop_if_char_run_ge=args.drop_if_char_run_ge,
        drop_if_rep_ratio_ge=args.drop_if_rep_ratio_ge,
        rep_ratio_min_tokens=args.rep_ratio_min_tokens,
    )
    stats = CleaningStats(
        source_rows_requested=OPENORCA_SUBSET_ROWS, source_revision=OPENORCA_REVISION
    )

    # We do 2-phase:
    # - filter + dedup while streaming
    # - reservoir-like deterministic sampling via hashing (no need to hold all rows)
    #
    # Approach:
    # - assign each kept row a stable pseudo-random score based on (seed, dedup_key)
    # - keep top-K by score using a small heap
    import heapq

    heap: list[Tuple[float, Dict[str, Any]]] = []
    seen_keys: set[bytes] = set()

    t0 = time.monotonic()
    last_t = t0

    for raw in iter_openorca_1m_subset():
        stats.seen_rows += 1
        try:
            q = str(raw.get("question", "") or "")
            r = str(raw.get("response", "") or "")
            if not _is_valid_sample(q, r, cfg, stats):
                continue

            key = _dedup_key(q, r)
            if key in seen_keys:
                stats.dropped_dedup += 1
                continue
            seen_keys.add(key)

            alpaca = to_alpaca_jsonl_row(raw)

            # Stable "random" score derived from key + seed (deterministic)
            score_payload = args.seed.to_bytes(8, "big", signed=False) + key
            score = int.from_bytes(hashlib.sha1(score_payload).digest(), "big") / float(2**160)

            if len(heap) < args.max_samples:
                heapq.heappush(heap, (score, alpaca))
            else:
                # keep larger scores (top-K)
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, alpaca))
        except Exception:
            stats.dropped_other += 1
            continue

        if args.progress_every and stats.seen_rows % args.progress_every == 0:
            now = time.monotonic()
            dt = now - last_t
            total_dt = now - t0
            rows_per_s = args.progress_every / dt if dt > 0 else None
            # Print a compact JSON line so `tail -f` is easy.
            print(
                json.dumps(
                    {
                        "progress": {
                            "seen_rows": stats.seen_rows,
                            "heap_size": len(heap),
                            "dedup_size": len(seen_keys),
                            "dropped_empty": stats.dropped_empty,
                            "dropped_length": stats.dropped_length,
                            "dropped_char_run": stats.dropped_char_run,
                            "dropped_repetition": stats.dropped_repetition,
                            "dropped_dedup": stats.dropped_dedup,
                            "dropped_other": stats.dropped_other,
                            "elapsed_min": round(total_dt / 60.0, 2),
                            "rows_per_sec": None if rows_per_s is None else round(rows_per_s, 2),
                        }
                    },
                    ensure_ascii=False,
                )
            )
            last_t = now

    # Finalize samples: sort by score descending for stable output
    heap.sort(key=lambda x: x[0], reverse=True)
    final_rows = [row for _, row in heap]

    # Write main output (recommend putting it under dataset_dir)
    os.makedirs(args.dataset_dir, exist_ok=True)
    out_path = args.out
    kept = write_jsonl(out_path, final_rows)
    stats.kept_rows = kept

    # Write dataset_info.json for LLaMA-Factory
    dataset_info_path = os.path.join(args.dataset_dir, "dataset_info.json")
    dataset_file_name = os.path.basename(out_path)
    # LLaMA-Factory resolves file_name under dataset_dir. If user writes `--out` elsewhere,
    # we mirror a copy into dataset_dir to keep training config simple.
    out_dir_abs = os.path.abspath(os.path.dirname(out_path))
    dataset_dir_abs = os.path.abspath(args.dataset_dir)
    if out_dir_abs != dataset_dir_abs:
        mirrored = os.path.join(args.dataset_dir, dataset_file_name)
        try:
            import shutil

            shutil.copy2(out_path, mirrored)
        except Exception:
            # If mirroring fails, training can still work if user sets dataset_dir accordingly.
            pass
    dataset_info = {
        args.dataset_name: {
            "file_name": dataset_file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
    }
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # Write stats
    os.makedirs(os.path.dirname(os.path.abspath(args.stats)), exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cleaning_config": asdict(cfg),
                "stats": asdict(stats),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Also write a small demo set (first 200 after sorting) for submission
    demo_n = min(200, len(final_rows))
    if args.demo_out:
        write_jsonl(args.demo_out, final_rows[:demo_n])
    if args.demo_dataset_info_out and args.demo_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.demo_dataset_info_out)), exist_ok=True)
        demo_info = {
            args.dataset_name: {
                "file_name": os.path.basename(args.demo_out),
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history",
                },
            }
        }
        with open(args.demo_dataset_info_out, "w", encoding="utf-8") as f:
            json.dump(demo_info, f, ensure_ascii=False, indent=2)

    # Print a short summary (useful when redirecting to logs)
    print(json.dumps({"kept_rows": kept, "seen_rows": stats.seen_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()

