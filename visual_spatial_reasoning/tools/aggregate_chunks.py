"""
Aggregate `question_chunk_*/results.json` files under a qc<N> directory
into a single top-level `results.json`. Optionally print a CSV row
with overall + per-qtype accuracy so a batch eval driver can collect
side-by-side comparisons.

Used by both manual aggregation and `scripts/batch_eval_ckpts.sh`.

Usage:
    python tools/aggregate_chunks.py path/to/_spatial_beam_search_qc8/
    python tools/aggregate_chunks.py path/to/_spatial_beam_search_qc8/ \
        --csv --label step60

The merge is per-question (concatenate correct/wrong qid lists), so
chunks of unequal size are weighted correctly. Re-running overwrites
the merged results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Tuple


# Canonical column order for CSV output. Matches the qtypes the test
# pipeline emits (note: test uses `action_conseq`, train uses
# `action_consequence` — keep the test-side spelling here).
_TEST_QTYPES = (
    "ego_movement", "obj_movement", "goal_aim", "action_conseq", "perspective",
)


def aggregate(root: str) -> Tuple[dict, int, int, list]:
    """Return (merged, total_correct, total_n, per_chunk).

    `per_chunk` is a list of `(name, done, total, all_acc)` for diagnostics.
    """
    chunks = sorted(
        d for d in os.listdir(root) if d.startswith("question_chunk_")
    )

    merged_progress = defaultdict(lambda: {"correct": [], "wrong": []})
    parsing = {"scores": 0, "answer": 0, "answer_qid": [], "scores_qid": []}
    skip_indices: list = []
    done = total = 0
    per_chunk: list = []

    for c in chunks:
        p = os.path.join(root, c, "results.json")
        if not os.path.exists(p):
            print(f"[warn] {p} missing", file=sys.stderr)
            continue
        with open(p) as f:
            d = json.load(f)
        a, b = (int(x.strip()) for x in d["current"].split("/"))
        done += a
        total += b
        per_chunk.append((c, a, b, d["accuracy"]["all"]))
        for qt, prog in d["progress"].items():
            merged_progress[qt]["correct"].extend(prog.get("correct", []))
            merged_progress[qt]["wrong"].extend(prog.get("wrong", []))
        pe = d.get("parsing_err_stats", {})
        parsing["scores"] += pe.get("scores", 0)
        parsing["answer"] += pe.get("answer", 0)
        parsing["answer_qid"].extend(pe.get("answer_qid", []))
        parsing["scores_qid"].extend(pe.get("scores_qid", []))
        skip_indices.extend(d.get("skip_indices", []))

    acc_types: dict = {}
    total_correct = total_n = 0
    for qt, prog in merged_progress.items():
        c = len(prog["correct"])
        w = len(prog["wrong"])
        n = c + w
        acc_types[qt] = (c / n) if n else 0.0
        total_correct += c
        total_n += n

    merged = {
        "current": f"{done} / {total}",
        "parsing_err_stats": parsing,
        "accuracy": {
            "all": (total_correct / total_n) if total_n else 0.0,
            "types": acc_types,
        },
        "skip_indices": sorted(set(skip_indices)),
        "progress": {qt: dict(p) for qt, p in merged_progress.items()},
    }

    out_path = os.path.join(root, "results.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged, total_correct, total_n, per_chunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="dir containing question_chunk_*/")
    p.add_argument("--csv", action="store_true",
                   help="print one CSV row to stdout: "
                        "label,total,acc_all,<per-qtype>...")
    p.add_argument("--label", default="",
                   help="first CSV column (e.g. checkpoint step number)")
    p.add_argument("--quiet", action="store_true",
                   help="skip the human-readable per-chunk / per-qtype dump")
    return p.parse_args()


def main():
    args = parse_args()
    merged, tc, tn, per_chunk = aggregate(args.root)
    types = merged["accuracy"]["types"]

    if not args.quiet:
        print(f"[merged] {merged['current']}  "
              f"acc={merged['accuracy']['all'] * 100:.2f}%  ({tc}/{tn})",
              file=sys.stderr)
        for c, a, b, acc in per_chunk:
            print(f"  {c}: {a:>4}/{b:<4} all={acc:.4f}", file=sys.stderr)
        for qt, acc in types.items():
            prog = merged["progress"][qt]
            cc = len(prog["correct"])
            ww = len(prog["wrong"])
            print(f"  {qt:20s} {acc * 100:6.2f}%  ({cc}/{cc + ww})",
                  file=sys.stderr)

    if args.csv:
        row = [
            str(args.label) if args.label else os.path.basename(args.root),
            f"{tc}/{tn}",
            f"{merged['accuracy']['all'] * 100:.2f}",
        ]
        for qt in _TEST_QTYPES:
            v = types.get(qt)
            row.append(f"{v * 100:.2f}" if v is not None else "")
        # Append any extra qtypes that weren't in the canonical list, so we
        # never silently drop data — these will land at the end of the row
        # and the header in the driver script needs to know about them.
        extras = sorted(qt for qt in types if qt not in _TEST_QTYPES)
        for qt in extras:
            row.append(f"{types[qt] * 100:.2f}")
        print(",".join(row))


if __name__ == "__main__":
    main()
