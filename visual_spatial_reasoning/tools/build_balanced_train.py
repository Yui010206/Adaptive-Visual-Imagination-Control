"""
Read the JSONL produced by tools/prescore_train_skip.py and emit a
balanced training-set JSON for GRPO. The bucketing is:

  easy_skip   gpt4o_skip_correct == True
  needs_wm    gpt4o_skip_correct == False
  drop        gpt4o_skip_correct is None  (errored during prescore)

You pick a target ratio (default 30/60/0 = 30% easy + 60% needs_wm
upsampled to ~equal, 0 hopeless since we can't identify those without WM).
The script repeats `needs_wm` entries with replacement to hit the ratio.

Run:
    python tools/build_balanced_train.py \
        --prescore data/train_prescored_10k.jsonl \
        --sample data/train_sample_10k.json \
        --output data/train_balanced_grpo.json \
        --easy_frac 0.3 --needs_wm_frac 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prescore", default="data/train_prescored_10k.jsonl",
                   help="JSONL output of tools/prescore_train_skip.py")
    p.add_argument("--sample", default="data/train_sample_10k.json",
                   help="The sampled-question JSON the prescore was run on. "
                        "Provides the full SAT entry (img_paths, choices, ...) "
                        "since prescore JSONL only stores qid/qtype/response.")
    p.add_argument("--output", default="data/train_balanced_grpo.json")
    p.add_argument("--easy_frac", type=float, default=0.30,
                   help="Target fraction of `easy_skip` in the output mix.")
    p.add_argument("--needs_wm_frac", type=float, default=0.70,
                   help="Target fraction of `needs_wm` in the output mix.")
    p.add_argument("--total", type=int, default=None,
                   help="Total size of the output set. Default: all needs_wm "
                        "kept once + easy_skip subsampled to match easy_frac.")
    p.add_argument("--seed", type=int, default=44)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ------------ Load prescore + full questions ------------
    label_by_qid: Dict[int, bool] = {}
    with open(args.prescore) as f:
        for line in f:
            r = json.loads(line)
            qid = r["qid"]
            v = r.get("gpt4o_skip_correct")
            if v is None:
                continue
            label_by_qid[qid] = bool(v)
    print(f"[prescore] {len(label_by_qid)} usable labels")

    with open(args.sample) as f:
        all_q = json.load(f)
    by_qid = {q["database_idx"]: q for q in all_q}

    easy = [by_qid[qid] for qid, v in label_by_qid.items()
            if v and qid in by_qid]
    hard = [by_qid[qid] for qid, v in label_by_qid.items()
            if not v and qid in by_qid]

    base_skip_acc = len(easy) / max(1, len(easy) + len(hard))
    print(f"[base] gpt4o-skip-only acc on prescore set: {base_skip_acc:.3f}  "
          f"(easy={len(easy)}, hard={len(hard)})")
    print(f"[base] qtype breakdown:")
    for label, bucket in [("easy_skip", easy), ("needs_wm", hard)]:
        ct = Counter(q["question_type"] for q in bucket)
        print(f"  {label:10s}: {dict(ct)}")

    # ------------ Compute target counts ------------
    if not (0 < args.easy_frac < 1 and 0 < args.needs_wm_frac < 1):
        raise ValueError("easy_frac and needs_wm_frac must each be in (0,1)")
    if abs(args.easy_frac + args.needs_wm_frac - 1.0) > 1e-6:
        raise ValueError("easy_frac + needs_wm_frac must sum to 1.0")

    if args.total is None:
        # Default: keep all `needs_wm` once, scale `easy_skip` so their
        # final ratio matches the requested mix. needs_wm is the scarce
        # signal-bearing bucket so we never throw any away.
        n_needs = len(hard)
        n_total = round(n_needs / args.needs_wm_frac)
        n_easy = n_total - n_needs
    else:
        n_total = args.total
        n_easy = round(n_total * args.easy_frac)
        n_needs = n_total - n_easy

    print(f"[target] total={n_total}  easy={n_easy}  needs_wm={n_needs}")

    # ------------ Sample / repeat ------------
    def take(pool: List[dict], n: int) -> List[dict]:
        if not pool:
            return []
        if n <= len(pool):
            rng.shuffle(pool)
            return pool[:n]
        # Upsample with replacement; keep order deterministic per seed.
        out = list(pool)
        while len(out) < n:
            out.append(rng.choice(pool))
        rng.shuffle(out)
        return out

    out = take(easy, n_easy) + take(hard, n_needs)
    rng.shuffle(out)

    # ------------ Write ------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[done] wrote {len(out)} questions to {args.output}")
    print(f"[done] mix: easy={sum(1 for q in out if label_by_qid.get(q['database_idx']))}  "
          f"needs_wm={sum(1 for q in out if not label_by_qid.get(q['database_idx']))}")
    print(f"[done] qtype: {dict(Counter(q['question_type'] for q in out))}")


if __name__ == "__main__":
    main()
