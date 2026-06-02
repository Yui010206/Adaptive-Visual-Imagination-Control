"""
Standalone tuner for Qwen3-VL policy sampling diversity.

Loads Qwen3-VL once, then for every (temperature, top_p, top_k) combo on the
CLI it samples K rollouts on each of `num_questions` real training questions
and reports how many UNIQUE policy plans came out. Goal: find sampling settings
where K=8 rollouts produce >=3-4 unique plans, otherwise GRPO has no signal.

Plan key = (decision, tuple of (atomic action type, value)).
The reason field is intentionally not part of the key — same plan, different
reason text still counts as the same plan (matches the training-time dedup).

Usage examples:
    # quick check at a single config
    python training/probe_policy_diversity.py --temperatures 1.2 --top_k_values 50

    # mini grid search (24 cells, K=8 each, on 4 questions)
    python training/probe_policy_diversity.py \
        --temperatures 0.7,1.0,1.2,1.5 \
        --top_p_values 1.0,0.95 \
        --top_k_values 0,50,100 \
        --num_questions 4 --K 8

The summary table at the end ranks configs by mean unique-plan count.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import List

# Repo-root on sys.path so we can import utils.* / etc when invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.prompt_formatting import format_spatial_vqa_prompt_policy_plan
from utils.qwen_policy import _content_tuples_to_qwen_user_blocks, _parse_policy_json


# ----------------------------------------------------------------------------
# Alternative "soft" policy prompt that asks for free-form reasoning before
# the JSON. Free-form text has higher entropy at every step than a strict
# JSON schema, which usually gives the model room to branch into different
# decisions. The parser still looks for a {...} block at the end so the
# downstream training loop doesn't need to change.
# ----------------------------------------------------------------------------
_SOFT_SYS_PROMPT = (
    "You are an assistant for spatial reasoning in a 3D indoor scene.\n"
    "Look at the image(s) and the multiple-choice question. You can either:\n"
    "  (a) answer directly from what you see (we'll skip the world model), OR\n"
    "  (b) call a world model to imagine new viewpoints (you provide actions).\n\n"
    "Calling the world model is expensive, so prefer (a) when the answer is\n"
    "already visible. Use (b) when you genuinely need a new viewpoint.\n\n"
    "Allowed atomic actions (type must be one of these strings):\n"
    "  - move-forward 0.25 meters\n"
    "  - turn-left 9 degrees\n"
    "  - turn-right 9 degrees\n"
    "Use repeated turns to approximate larger angles (e.g., 27° = 3 turns).\n\n"
    "RESPONSE FORMAT — strict, anything else makes the parser fail:\n"
    "  Line 1: ONE short sentence of reasoning, <= 25 words, English only.\n"
    "  Line 2: a single JSON object on one line, no markdown fences, no\n"
    "          extra text after it.\n"
    "Allowed `decision` values: \"skip\" or \"call_wm\" (lowercase, exactly).\n"
    "Allowed `type` values:     \"move-forward\", \"turn-left\", \"turn-right\".\n"
    "Concrete examples (copy this format exactly):\n"
    '  Easy:  The lamp is plainly visible on the right side of the desk.\n'
    '         {"decision": "skip", "actions": []}\n'
    '  Hard:  The chair is occluded; turning a bit to see behind the bed.\n'
    '         {"decision": "call_wm", "actions": [{"type": "turn-left", "value": 18}]}\n'
    '  Multi: Need to step forward and look right to see the table edge.\n'
    '         {"decision": "call_wm", "actions": [{"type": "move-forward", "value": 0.5}, {"type": "turn-right", "value": 27}]}\n'
    "Rules: For skip, actions MUST be []. For call_wm, give 1-6 actions, "
    "every `value` must be a positive number.\n"
)


def _build_soft_user_blocks(question, answer_choices, images):
    """Soft prompt with a hard FORMAT anchor at the tail. The "Output:" line
    deliberately ends mid-stream so the model continues by emitting the JSON
    immediately, instead of drifting into a multi-paragraph CoT that hits
    the 512-token cap before any JSON is produced."""
    blocks = []
    for i, p in enumerate(images):
        blocks.append({"type": "text", "text": f"Image {i + 1}:"})
        blocks.append({"type": "image", "image": p})
    text = (
        f"\nQuestion: {question}\n"
        f"Answer choices: {answer_choices}\n\n"
        f"Reasoning (one short sentence, <=25 words): "
    )
    blocks.append({"type": "text", "text": text})
    return blocks


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct",
                   help="single model id (back-compat). For multi-model probe, "
                        "use --models 'name1,name2,...'.")
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated list of model ids to probe. "
                        "Overrides --model_name when set.")
    p.add_argument("--input_file", default="data/train.json")
    p.add_argument("--num_questions", type=int, default=4)
    p.add_argument("--K", type=int, default=8,
                   help="rollouts per (config, question) cell")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_images", type=int, default=2)

    p.add_argument("--prompt_style", choices=["strict", "soft"], default="strict",
                   help="strict: existing JSON-only policy_plan prompt. "
                        "soft: free-form reasoning + JSON tail. Soft prompts "
                        "usually have far higher token entropy.")

    p.add_argument("--temperatures", type=str, default="0.7,1.0,1.2,1.5",
                   help="comma-separated list of temperatures to try")
    p.add_argument("--top_p_values", type=str, default="1.0,0.95",
                   help="comma-separated list of top_p values; use 1.0 to disable")
    p.add_argument("--top_k_values", type=str, default="0,50,100",
                   help="comma-separated list of top_k values; 0 disables top_k")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help=">1.0 penalizes repeating tokens (helps when the model "
                        "loops the same JSON shape across rollouts).")

    p.add_argument("--print_plans", action="store_true", default=True,
                   help="print every unique plan per (config, question)")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; each (config, question) cell uses seed + cell_idx "
                        "for reproducibility. K rollouts inside one generate call are "
                        "still drawn independently via num_return_sequences.")
    p.add_argument("--show_entropy", action="store_true", default=True,
                   help="for the first question of each config, print the entropy + "
                        "top-5 probabilities of the first generated token. If top-1 "
                        "p > 0.95 even at high T, the model is too confident -> "
                        "no amount of seed-rotation will diversify samples.")
    p.add_argument("--separate_calls", action="store_true", default=False,
                   help="instead of one generate(num_return_sequences=K), do K "
                        "separate generate(num_return_sequences=1) calls each with "
                        "its own seed. Useful to rule out batched-sampling collapse.")
    return p.parse_args()


def load_questions(path: str, n: int, max_images: int, seed: int) -> List[dict]:
    with open(path, "r") as f:
        all_q = json.load(f)
    filtered = [
        q for q in all_q
        if q.get("question_type") not in ("other",)
        and len(q.get("img_paths", [])) <= max_images
    ]
    import random
    random.Random(seed).shuffle(filtered)
    return filtered[:n]


def build_messages(question, prompt_style="strict"):
    images = question["img_paths"][:2]
    if prompt_style == "soft":
        return [
            {"role": "system",
             "content": [{"type": "text", "text": _SOFT_SYS_PROMPT}]},
            {"role": "user",
             "content": _build_soft_user_blocks(
                 question["question"], question["answer_choices"], images)},
        ]
    # default: strict policy_plan prompt (same as training)
    sys_prompt, content = format_spatial_vqa_prompt_policy_plan(
        question=question["question"],
        answer_choices=question["answer_choices"],
        images=images,
    )
    user_blocks = _content_tuples_to_qwen_user_blocks(content)
    return [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": user_blocks},
    ]


def plan_signature(parsed):
    """Hashable plan key. None if parse failed."""
    if not parsed["parse_ok"]:
        return None
    return (
        parsed["decision"],
        tuple(
            (a["type"], round(float(a["value"]), 4))
            for a in parsed["atomic_actions"]
        ),
    )


def short_plan_str(parsed):
    if not parsed["parse_ok"]:
        return "PARSE_FAIL"
    if parsed["decision"] == "skip":
        return "skip"
    actions = ", ".join(
        f"{a['type']}={a['value']}" for a in parsed["raw_actions"]
    ) or "(none)"
    return f"call_wm[{actions}]"


@torch.no_grad()
def sample_K(
    model, processor, messages, K, temperature, top_p, top_k, max_new_tokens,
    seed=None, return_entropy=False, repetition_penalty=1.0,
):
    """
    Returns (texts, entropy_info) where entropy_info is None unless
    return_entropy=True. entropy_info = dict with keys:
        first_token_entropy : entropy of next-token distribution AT temp=1.0
        first_token_entropy_at_T : same but at the requested temperature
        top5_at_T : list of (token_str, prob) for top-5 candidates at temp T
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    entropy_info = None
    if return_entropy:
        # one no-grad forward to get raw logits at the first generation step
        out = model(**inputs, use_cache=False)
        last_logits = out.logits[0, -1, :].float()  # (V,)
        # raw (T=1.0)
        p_raw = F.softmax(last_logits, dim=-1)
        ent_raw = float(-(p_raw * torch.log(p_raw.clamp_min(1e-12))).sum())
        # at the requested T
        p_T = F.softmax(last_logits / max(temperature, 1e-6), dim=-1)
        ent_T = float(-(p_T * torch.log(p_T.clamp_min(1e-12))).sum())
        topv, topi = torch.topk(p_T, k=5)
        tok = processor.tokenizer
        top5 = [(repr(tok.decode([int(i)])), float(v))
                for v, i in zip(topv.tolist(), topi.tolist())]
        entropy_info = {
            "first_token_entropy_T1": ent_raw,
            "first_token_entropy_atT": ent_T,
            "top5_at_T": top5,
        }

    gen_kwargs = dict(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_return_sequences=K,
    )
    if top_p is not None and top_p < 1.0:
        gen_kwargs["top_p"] = top_p
    if top_k is not None and top_k > 0:
        gen_kwargs["top_k"] = top_k
    if repetition_penalty is not None and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    gen = out[:, prompt_len:]
    texts = processor.batch_decode(
        gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return texts, entropy_info


def _load_model(model_name):
    """Load any HF image-text-to-text VLM (covers Qwen3VL, Qwen2.5VL, etc.)."""
    print(f"Loading {model_name} ...", flush=True)
    # Single-GPU load: device_map="auto" shards across multiple GPUs and
    # triggers a transformers bug where the broadcasted bool mask in
    # Qwen2.5VL.get_placeholder_mask gets corrupted after the cross-device
    # `.to()` (sum drops to 0, raising "Image features and image tokens do
    # not match" even when shapes match). 7B/4B fits on one A100 80GB.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype="auto", device_map=device, trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def _free_model(model):
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    temps = [float(x) for x in args.temperatures.split(",")]
    top_ps = [float(x) for x in args.top_p_values.split(",")]
    top_ks = [int(x) for x in args.top_k_values.split(",")]
    configs = [(t, p, k) for t in temps for p in top_ps for k in top_ks]

    model_names = (
        [s.strip() for s in args.models.split(",")] if args.models
        else [args.model_name]
    )

    print(f"Loading questions from {args.input_file} ...", flush=True)
    questions = load_questions(
        args.input_file, args.num_questions,
        max_images=args.max_images, seed=args.seed,
    )
    print(f"  {len(questions)} questions selected.", flush=True)
    for q in questions:
        print(f"  qid={q['database_idx']:>6} ({q['question_type']:<22}) "
              f"choices={len(q['answer_choices'])}", flush=True)

    summary_all = []  # (model, config) -> row

    for model_name in model_names:
        model, processor = _load_model(model_name)
        print(f"\n##############  MODEL: {model_name}  "
              f"(prompt_style={args.prompt_style})  ##############", flush=True)
        summary = run_probe_for_one_model(
            args, model, processor, questions, configs,
        )
        for row in summary:
            row["model"] = model_name
            summary_all.append(row)
        _free_model(model)

    # ---- Final ranking across all models ----
    print(f"\n{'='*88}\nFINAL RANKING (sorted by avg unique plans, desc)\n{'='*88}",
          flush=True)
    print(f"{'model':<32} {'config':<24} {'avg_uniq':>9}  "
          f"{'skip%':>7}  {'call%':>7}  {'pf%':>7}", flush=True)
    for row in sorted(summary_all, key=lambda r: -r["avg_unique"]):
        print(
            f"{row['model']:<32} "
            f"{row['config']:<24} "
            f"{row['avg_unique']:>9.2f}  "
            f"{row['skip_rate']:>6.1%}  "
            f"{row['call_rate']:>6.1%}  "
            f"{row['parse_fail_rate']:>6.1%}",
            flush=True,
        )


def run_probe_for_one_model(args, model, processor, questions, configs):
    summary = []
    for cfg_i, (temp, top_p, top_k) in enumerate(configs):
        cfg_str = f"T={temp}_P={top_p}_K={top_k}"
        print(f"\n{'='*72}\n[config {cfg_i + 1}/{len(configs)}] {cfg_str}\n{'='*72}",
              flush=True)
        per_q_unique = []
        per_q_decision = []
        per_q_parse_fail = []
        for q_i, q in enumerate(questions):
            messages = build_messages(q, prompt_style=args.prompt_style)
            cell_seed = args.seed + cfg_i * 10_000 + q_i
            want_entropy = args.show_entropy and q_i == 0
            texts, ent = sample_K(
                model, processor, messages, args.K,
                temperature=temp, top_p=top_p, top_k=top_k,
                max_new_tokens=args.max_new_tokens,
                seed=cell_seed,
                return_entropy=want_entropy,
                repetition_penalty=args.repetition_penalty,
            )
            if ent is not None:
                print(f"  [first-token entropy on qid={q['database_idx']}] "
                      f"H(T=1.0)={ent['first_token_entropy_T1']:.3f} nats   "
                      f"H(T={temp})={ent['first_token_entropy_atT']:.3f} nats",
                      flush=True)
                print(f"     top-5 next-tokens at T={temp}:", flush=True)
                for tok_repr, p in ent["top5_at_T"]:
                    print(f"       {p:6.3f}  {tok_repr}", flush=True)
            sigs, decisions, parsed_list = [], [], []
            for t in texts:
                parsed = _parse_policy_json(
                    t, interval_meter=0.25, interval_angle=9, max_atomic=6,
                )
                parsed_list.append(parsed)
                sigs.append(plan_signature(parsed))
                if parsed["parse_ok"]:
                    decisions.append(parsed["decision"])
                else:
                    decisions.append("PARSE_FAIL")

            n_unique = len(set(s for s in sigs if s is not None))
            n_pf = sum(1 for d in decisions if d == "PARSE_FAIL")
            counts = Counter(decisions)
            per_q_unique.append(n_unique)
            per_q_decision.append(counts)
            per_q_parse_fail.append(n_pf)

            print(f"  qid={q['database_idx']:>6}  "
                  f"unique_plans={n_unique}/{args.K}  "
                  f"skip={counts.get('skip', 0)}  "
                  f"call_wm={counts.get('call_wm', 0)}  "
                  f"parse_fail={n_pf}", flush=True)

            if args.print_plans:
                shown = set()
                for j, parsed in enumerate(parsed_list):
                    sig = plan_signature(parsed)
                    if sig in shown and sig is not None:
                        continue
                    shown.add(sig)
                    print(f"    [{j}] {short_plan_str(parsed)}", flush=True)

        avg_unique = sum(per_q_unique) / max(1, len(per_q_unique))
        avg_pf = sum(per_q_parse_fail) / max(1, len(per_q_parse_fail))
        skip_rate = sum(c.get("skip", 0) for c in per_q_decision) / (
            args.K * max(1, len(questions))
        )
        call_rate = sum(c.get("call_wm", 0) for c in per_q_decision) / (
            args.K * max(1, len(questions))
        )
        summary.append({
            "config": cfg_str,
            "T": temp, "top_p": top_p, "top_k": top_k,
            "avg_unique": avg_unique,
            "skip_rate": skip_rate,
            "call_rate": call_rate,
            "parse_fail_rate": avg_pf / args.K,
        })
        print(f"  -> avg unique={avg_unique:.2f}/{args.K}  "
              f"skip%={skip_rate:.2%}  call%={call_rate:.2%}  "
              f"parse_fail%={avg_pf / args.K:.2%}", flush=True)

    # ---- Per-model summary ----
    print(f"\n{'='*72}\nSUMMARY (this model, sorted by avg unique plans, desc)\n{'='*72}",
          flush=True)
    print(f"{'config':<28} {'avg_unique':>10}  {'skip%':>7}  {'call%':>7}  {'pf%':>7}",
          flush=True)
    for row in sorted(summary, key=lambda r: -r["avg_unique"]):
        print(
            f"{row['config']:<28} "
            f"{row['avg_unique']:>10.2f}  "
            f"{row['skip_rate']:>6.1%}  "
            f"{row['call_rate']:>6.1%}  "
            f"{row['parse_fail_rate']:>6.1%}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    main()
