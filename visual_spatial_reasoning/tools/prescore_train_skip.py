"""
Prescore SAT-train questions with GPT-4o under a "skip-only" baseline (no
world model). The output (one JSON line per question) feeds the GRPO data
balancer: questions where GPT-4o is right from the original image alone are
"easy_skip" and should be downsampled; questions GPT-4o gets wrong are
"needs_wm" and should be upsampled, since those are the ones with real
GRPO advantage signal.

Pipeline:
  1) Stratified sample of `--num_questions` from data/train.json
     (drops question_type="other"; quotas per type, per num_images).
  2) Multi-threaded GPT-4o calls using utils.api.ChatAPI.
  3) Append-only JSONL output, resumable: qids already in the file are
     skipped.

Run from repo root with conda env `avic`:
    python tools/prescore_train_skip.py \
        --num_questions 10000 \
        --concurrency 8 \
        --output data/train_prescored_10k.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Ensure repo root on sys.path so we can import utils.* when invoked from
# repo root or from elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.api import AzureConfig, ChatAPI
from utils.prompt_formatting import (
    format_gpt_content,
    format_spatial_vqa_prompt_answer_baseline,
)


# Question types that the trainer actually uses (matches the
# `question_type in ("other",): continue` filter in train_qwen_grpo.py).
USEFUL_TYPES = (
    "action_consequence",
    "action_sequence",
    "obj_movement",
    "goal_aim",
    "perspective",
)


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="data/train.json")
    p.add_argument("--output", default="data/train_prescored_5k.jsonl",
                   help="JSONL: one line per question with gpt4o_skip_correct.")
    p.add_argument("--sample_file", default="data/train_sample_5k.json",
                   help="Where to write the (deterministic) sampled subset. "
                        "Re-running the script with the same seed produces "
                        "the same sample; if this file already exists it is "
                        "loaded as-is so resume-after-crash is bit-exact.")
    p.add_argument("--num_questions", type=int, default=5000)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Parallel GPT-4o requests in flight.")
    p.add_argument("--max_retries", type=int, default=5,
                   help="Per-question retries on rate-limit / transient errors.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--api_version", default="2024-12-01-preview")
    p.add_argument("--qtypes", default=",".join(USEFUL_TYPES),
                   help="Comma-separated qtypes to include.")
    p.add_argument("--image_hash_cache",
                   default="data/.image_md5_cache.json",
                   help="Disk cache of (resolved-path -> md5) so we don't "
                        "rehash the train set on every run.")
    p.add_argument("--hash_workers", type=int, default=16,
                   help="Threads used for the one-time image-hash pass.")
    return p.parse_args()


# ----------------------------------------------------------------------------
def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_image_hashes(
    questions: List[dict],
    cache_path: str,
    workers: int = 16,
) -> Dict[str, str]:
    """Return path -> md5(content). Persists/loads from JSON cache so
    repeated runs are fast. The dataset has ~40k unique paths but only
    ~50% of those are unique on content (different filenames for the
    same scene image), which is the whole reason we hash."""
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            print(f"[hash] loaded {len(cache)} cached hashes from {cache_path}")
        except Exception:
            cache = {}

    needed = sorted({resolve_image(p) for q in questions for p in q.get("img_paths", [])
                     if resolve_image(p) not in cache})
    if not needed:
        return cache

    print(f"[hash] computing md5 for {len(needed)} new images...")
    t0 = time.time()
    lock = threading.Lock()
    done = [0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_md5_file, p): p for p in needed}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                h = fut.result()
            except FileNotFoundError:
                # Skip; sample step will drop questions whose images
                # we couldn't read.
                h = ""
            with lock:
                cache[p] = h
                done[0] += 1
                if done[0] % 5000 == 0 or done[0] == len(needed):
                    print(f"[hash] {done[0]}/{len(needed)}  "
                          f"({done[0] / (time.time() - t0):.0f}/s)")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)
    print(f"[hash] cache -> {cache_path} ({len(cache)} entries)")
    return cache


def _question_image_key(q: dict, hashes: Dict[str, str]) -> Tuple[str, ...]:
    """Image-content fingerprint of a question = sorted tuple of md5s
    over all its img_paths. Two questions sharing this key reuse the
    *exact same scene* (and thus add no visual diversity)."""
    return tuple(sorted(hashes.get(resolve_image(p), "") for p in q.get("img_paths", [])))


def stratified_sample(
    questions: List[dict],
    qtypes: List[str],
    target: int,
    seed: int,
    image_hashes: Dict[str, str],
) -> List[dict]:
    """Equal per-qtype quotas (`target // len(qtypes)`, capped by
    availability), with the leftover redistributed to types that still
    have headroom. Within each type, group by image-content hash and
    round-robin through the groups so we exhaust unique scenes before
    reusing any.

    The dataset has heavy image reuse (one render is paired with
    multiple questions). Naive random sampling pulls many questions
    sharing the same image, which wastes prescore budget — same image
    means GPT-4o's skip-only correctness is highly correlated across
    those questions. Round-robin over image groups breaks that.
    """
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for q in questions:
        if q.get("question_type") in qtypes:
            by_type[q["question_type"]].append(q)

    total_avail = sum(len(v) for v in by_type.values())
    if target > total_avail:
        print(f"[sample] target {target} > available {total_avail}; "
              f"using all available")
        target = total_avail

    # Equal quotas, capped by availability, residual redistributed.
    n_types = len(by_type)
    base = target // n_types
    quotas = {t: min(base, len(by_type[t])) for t in by_type}
    deficit = target - sum(quotas.values())
    while deficit > 0:
        growable = [t for t in quotas if quotas[t] < len(by_type[t])]
        if not growable:
            break
        # Round-robin add 1 to types with the most headroom.
        growable.sort(key=lambda t: len(by_type[t]) - quotas[t], reverse=True)
        for t in growable:
            if deficit <= 0:
                break
            quotas[t] += 1
            deficit -= 1

    print(f"[sample] per-qtype quotas: "
          f"{ {t: f'{n}/{len(by_type[t])}' for t, n in quotas.items()} }")

    rng = random.Random(seed)
    sampled: List[dict] = []
    for t, n in quotas.items():
        # Group by image-content key.
        groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
        for q in by_type[t]:
            groups[_question_image_key(q, image_hashes)].append(q)
        # Shuffle inside each group; shuffle group order too.
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        for k in group_keys:
            rng.shuffle(groups[k])

        n_unique_imgs_total = len(groups)
        n_questions_total = len(by_type[t])

        # Round-robin: round 0 = one question from each group, etc.
        kept: List[dict] = []
        round_idx = 0
        while len(kept) < n:
            took_any = False
            for k in group_keys:
                if len(kept) >= n:
                    break
                if round_idx < len(groups[k]):
                    kept.append(groups[k][round_idx])
                    took_any = True
            if not took_any:
                break
            round_idx += 1
        sampled.extend(kept)

        # Diversity report: how many unique image-sets did we hit?
        unique_imgs_hit = len({_question_image_key(q, image_hashes) for q in kept})
        print(f"  {t:20s}: kept {len(kept):>5}  "
              f"unique-images {unique_imgs_hit}/{n_unique_imgs_total}  "
              f"(of {n_questions_total} questions)")

    rng.shuffle(sampled)
    return sampled


# ----------------------------------------------------------------------------
def resolve_image(path: str) -> str:
    """Question img_paths are typically `./data/...`. Make absolute against
    the repo root so the script doesn't depend on CWD."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


def is_correct(response: str, question: dict) -> bool:
    """Match the rule used by pipelines/pipeline_baseline.py:_process_answer:
    the chosen answer is the last line, and we say `correct` iff the GT
    string appears as substring of that last line."""
    if not response:
        return False
    last_line = response.strip().split("\n")[-1].lower()
    gt = question["correct_answer"].lower()
    return gt in last_line


def call_gpt4o_skip(api: ChatAPI, q: dict) -> dict:
    """One QA call with the same baseline prompt the test pipeline uses,
    feeding only the original images (= skip-decision rollout)."""
    images = [resolve_image(p) for p in q["img_paths"]]
    sys_prompt, content_tuples = format_spatial_vqa_prompt_answer_baseline(
        question=q["question"],
        answer_choices=q["answer_choices"],
        images=images,
    )
    content = format_gpt_content(content_tuples)
    resp = api.get_system_response_with_content(sys_prompt, content)
    return {
        "qid": q["database_idx"],
        "qtype": q["question_type"],
        "n_images": len(q.get("img_paths", [])),
        "correct_answer": q["correct_answer"],
        "gpt4o_response": resp,
        "gpt4o_skip_correct": is_correct(resp, q),
    }


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    qtypes = [t.strip() for t in args.qtypes.split(",") if t.strip()]

    # ------------ Sample (or load) ------------
    if os.path.exists(args.sample_file):
        with open(args.sample_file) as f:
            sampled = json.load(f)
        print(f"[sample] loaded {len(sampled)} from existing {args.sample_file}")
    else:
        with open(args.train_file) as f:
            all_q = json.load(f)
        print(f"[sample] {len(all_q)} total questions in {args.train_file}")
        # Hash images on the candidate pool only (drop "other" first to
        # keep the hashing pass small and avoid a 130k cache blow-up).
        candidates = [q for q in all_q if q.get("question_type") in qtypes]
        print(f"[sample] {len(candidates)} candidates after qtype filter")
        image_hashes = _build_image_hashes(
            candidates, args.image_hash_cache, workers=args.hash_workers)
        sampled = stratified_sample(
            candidates, qtypes, args.num_questions, args.seed,
            image_hashes=image_hashes,
        )
        os.makedirs(os.path.dirname(args.sample_file) or ".", exist_ok=True)
        with open(args.sample_file, "w") as f:
            json.dump(sampled, f, indent=2)
        print(f"[sample] wrote {len(sampled)} to {args.sample_file}")
    print(f"[sample] qtype distribution: "
          f"{dict(Counter(q['question_type'] for q in sampled))}")

    # ------------ Resume: load already-scored qids ------------
    done_qids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_qids.add(json.loads(line)["qid"])
                except Exception:
                    pass
        print(f"[resume] {len(done_qids)} qids already in {args.output}")

    todo = [q for q in sampled if q["database_idx"] not in done_qids]
    if not todo:
        print("[resume] nothing to do; all questions already scored")
        return
    print(f"[run] {len(todo)} questions remaining")

    # ------------ Build a thread-local API client ------------
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if not azure_key or not azure_endpoint:
        print("[fatal] AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT must be set",
              file=sys.stderr)
        sys.exit(1)

    _local = threading.local()

    def get_api() -> ChatAPI:
        api = getattr(_local, "api", None)
        if api is None:
            cfg = AzureConfig(
                args.model,
                api_version=args.api_version,
                provider="azure",
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
            )
            api = ChatAPI(cfg)
            _local.api = api
        return api

    # ------------ Worker with retry/backoff ------------
    def worker(q):
        last_err = None
        for attempt in range(args.max_retries):
            try:
                return call_gpt4o_skip(get_api(), q)
            except Exception as e:
                last_err = e
                # Exponential backoff with jitter; longer for rate-limit errors.
                msg = str(e)
                base = 8.0 if "429" in msg or "rate" in msg.lower() else 2.0
                sleep_s = base * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(min(sleep_s, 60.0))
        return {
            "qid": q["database_idx"],
            "qtype": q["question_type"],
            "n_images": len(q.get("img_paths", [])),
            "correct_answer": q["correct_answer"],
            "gpt4o_response": None,
            "gpt4o_skip_correct": None,
            "error": f"{type(last_err).__name__}: {last_err}" if last_err else "unknown",
        }

    # ------------ Run with threads + serialised JSONL writes ------------
    write_lock = threading.Lock()
    n_done = 0
    n_correct = 0
    n_err = 0
    t0 = time.time()
    with open(args.output, "a") as fout, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(worker, q) for q in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with write_lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            n_done += 1
            if rec.get("gpt4o_skip_correct") is True:
                n_correct += 1
            if rec.get("error"):
                n_err += 1
            if n_done % 50 == 0 or n_done == len(todo):
                rate = n_done / max(1.0, time.time() - t0)
                eta = (len(todo) - n_done) / rate / 60 if rate else float("inf")
                print(f"[progress] {n_done}/{len(todo)}  "
                      f"acc={n_correct / max(1, n_done - n_err):.3f}  "
                      f"err={n_err}  rate={rate:.2f}/s  eta={eta:.1f}min",
                      flush=True)

    # ------------ Summary ------------
    print(f"[done] wrote {n_done} new lines to {args.output} "
          f"(skip-acc={n_correct / max(1, n_done - n_err):.3f}, errors={n_err})")


if __name__ == "__main__":
    main()
