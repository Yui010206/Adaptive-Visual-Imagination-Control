"""
Online GRPO training for the Qwen3-VL-4B policy.

Per-step flow on one rank:
  1. Pick `--per_device_batch_size` questions from the train shard.
  2. For each question:
       a. Build the policy prompt (same format as utils/qwen_policy.py uses).
       b. Sample K rollouts from the LoRA-wrapped policy (high temperature).
       c. Parse each rollout into a plan; reward each by running WM (when
          applicable) + GPT-4o QA (training/wm_qa_rollout.py).
  3. Compute group-relative advantages (per question, across the K rollouts).
  4. For every rollout, compute the GRPO loss and backward.
  5. After `--grad_accum_steps` micro-batches, optimizer.step().
  6. Save LoRA adapter every `--save_every` steps (rank 0 only).

Launch with `torchrun --nproc_per_node 8 training/train_qwen_grpo.py ...`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List, Optional

# IMPORTANT: must run BEFORE `import torch`. Under torchrun each child gets a
# distinct LOCAL_RANK; we narrow CUDA_VISIBLE_DEVICES so each rank sees only
# its own GPU (mapped to cuda:0). Necessary because the SVC world model
# hard-codes some buffers to cuda:0 and would mix devices on rank 1+.
if "LOCAL_RANK" in os.environ:
    _lr = int(os.environ["LOCAL_RANK"])
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _vis:
        _ids = [s for s in _vis.split(",") if s != ""]
        if _lr < len(_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = _ids[_lr]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_lr)

# Make sure the repo root is importable when launched from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# `pipelines/` is a sibling that imports its modules without a package prefix.
_PIPELINES_DIR = os.path.join(_REPO_ROOT, "pipelines")
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

import torch
import torch.distributed as dist

from utils.prompt_formatting import format_spatial_vqa_prompt_policy_plan
from utils.qwen_policy import _content_tuples_to_qwen_user_blocks, _parse_policy_json
# Soft-prompt builders are defined in the probe script; we reuse them here so
# `--prompt_style soft` matches what the probe characterized (much higher
# token entropy than the strict JSON-only policy_plan prompt).
from training.probe_policy_diversity import (
    _SOFT_SYS_PROMPT,
    _build_soft_user_blocks,
)

from training.qwen_grpo import QwenGRPOTrainer, compute_group_advantages
from training.wm_qa_rollout import WMQARollout, compute_reward, prepare_image_pair


# ----------------------------------------------------------------------------
#  Args
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_file", type=str, required=True,
                   help="JSON file with the questions (e.g. data/train.json).")
    p.add_argument("--scratch_dir", type=str, default="/tmp/grpo_scratch",
                   help="Per-rank workspace for intermediate WM artifacts.")

    # Policy
    p.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # QA model (used only at rollout time, not trained)
    p.add_argument("--qa_model_name", type=str, default="gpt-4o")
    p.add_argument("--qa_provider", type=str, default="azure")

    # GRPO
    p.add_argument("--num_rollouts", type=int, default=8,
                   help="K — number of rollouts per question per step.")
    p.add_argument("--rollout_temperature", type=float, default=1.2)
    p.add_argument("--rollout_top_p", type=float, default=0.95)
    p.add_argument("--rollout_top_k", type=int, default=50)
    p.add_argument("--rollout_max_new_tokens", type=int, default=512)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.04)

    # Reward
    p.add_argument("--action_cost", type=float, default=0.1,
                   help="Reward subtracted per atomic action.")
    p.add_argument("--parse_fail_penalty", type=float, default=-0.5)
    p.add_argument("--skip_wrong_penalty", type=float, default=0.0,
                   help="Extra penalty subtracted when the policy chose "
                        "`skip` AND got the answer wrong. Defaults to 0 "
                        "(legacy behaviour). Setting >0 makes a wrong skip "
                        "strictly worse than a wrong call_wm, which fights "
                        "skip-collapse: the model has incentive to call WM "
                        "when it isn't confident.")

    # Optim
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--per_device_batch_size", type=int, default=1,
                   help="Questions per micro-batch per GPU.")
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=2000,
                   help="Total optimizer steps.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_atomic_actions", type=int, default=6,
                   help="Cap atomic actions in the parsed plan (matches inference cap).")
    p.add_argument("--sampling_interval_meter", type=float, default=0.25)
    p.add_argument("--sampling_interval_angle", type=int, default=9)

    # WM args (mirror the SVC defaults used by scripts/pipeline_avic.sh)
    p.add_argument("--task", type=str, default="img2trajvid_s-prob")
    p.add_argument("--replace_or_include_input", type=bool, default=True)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--guider", type=int, default=1)
    p.add_argument("--L_short", type=int, default=576)
    p.add_argument("--num_targets", type=int, default=8)
    p.add_argument("--use_traj_prior", type=bool, default=True)
    p.add_argument("--chunk_strategy", type=str, default="interp")
    p.add_argument("--frame_interval", type=int, default=3)
    p.add_argument("--max_action_ids_cap", type=int, default=6)
    p.add_argument("--max_images", type=int, default=2)

    # Bookkeeping
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Verbose logging of every rollout (plan text + QA response + reward).
    p.add_argument("--verbose_rollouts", action="store_true", default=True)
    p.add_argument("--no_verbose_rollouts", dest="verbose_rollouts",
                   action="store_false")
    p.add_argument("--print_plan_chars", type=int, default=400,
                   help="Truncate the printed Qwen plan text to this many chars.")
    p.add_argument("--print_qa_chars", type=int, default=300,
                   help="Truncate the printed GPT-4o QA response to this many chars.")
    p.add_argument("--keep_rollout_dirs", action="store_true", default=False,
                   help="Skip cleanup of per-rollout scratch dirs. Useful for "
                        "inspecting WM outputs (pred.mp4, sampled frames) but "
                        "blows up disk fast (~50MB/rollout x 1024 rollouts/step).")

    # KL safety net: hard early-stop if running mean KL stays above threshold.
    # Healthy LoRA-GRPO sits in 0.05-0.3 nat. Previous failed run hit 0.1 by
    # micro-batch 70 and kept climbing, so stop early.
    p.add_argument("--kl_early_stop_threshold", type=float, default=0.5,
                   help="Abort training if a rolling mean of loss/kl exceeds "
                        "this value (in nats). 0.5 is the upper edge of healthy; "
                        "above this the policy is starting to drift hard.")
    p.add_argument("--kl_early_stop_window", type=int, default=10,
                   help="Window size (in micro-batches) for the rolling mean "
                        "used by --kl_early_stop_threshold.")

    # Wandb (rank 0 only)
    p.add_argument("--wandb_project", type=str, default=None,
                   help="If set, log metrics to this wandb project. "
                        "Combine with --wandb_entity for entity/project form.")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument(
        "--prompt_style",
        choices=["strict", "soft"],
        default="strict",
        help="strict: JSON-only policy_plan prompt (matches inference). "
             "soft: free-form reasoning + JSON tail; far higher token entropy, "
             "needed when K rollouts collapse to a single plan under strict.",
    )

    # Resume from a previously saved LoRA adapter. We don't persist optimizer
    # state, so resuming starts AdamW fresh — fine for GRPO where each step is
    # already largely independent (per-question rollouts), but means the LR
    # schedule effectively restarts.
    p.add_argument("--resume_adapter", type=str, default=None,
                   help="Path to a saved LoRA adapter dir (e.g. "
                        "nips_results/.../adapter_step140) to resume from. "
                        "When set, LoRA weights are loaded from this dir "
                        "instead of being initialised fresh.")
    p.add_argument("--start_step", type=int, default=0,
                   help="Initial value of the step counter. Set to the step "
                        "of the resumed adapter so saves continue with the "
                        "right tag (e.g. resume from adapter_step140 -> "
                        "next save will be adapter_step150).")

    return p.parse_args()


# ----------------------------------------------------------------------------
#  DDP helpers
# ----------------------------------------------------------------------------
def setup_ddp():
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not is_distributed:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    # We remapped CUDA_VISIBLE_DEVICES at module top so each rank only sees
    # its own GPU (as cuda:0). Always set device 0 here.
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl")
    return rank, world, local


def is_main(rank: int) -> bool:
    return rank == 0


def log(msg: str, rank: int):
    print(f"[rank{rank}] {msg}", flush=True)


# ----------------------------------------------------------------------------
#  Question filtering (mirror the inference filter)
# ----------------------------------------------------------------------------
def load_questions(train_file: str, max_images: int) -> list:
    with open(train_file, "r") as f:
        all_q = json.load(f)
    out = []
    for q in all_q:
        if q.get("question_type") in ("other",):
            continue
        if len(q.get("img_paths", [])) > max_images:
            continue
        out.append(q)
    return out


# ----------------------------------------------------------------------------
#  Build the policy prompt for one question (matches utils/qwen_policy.py)
# ----------------------------------------------------------------------------
def build_policy_messages(question, primary_img, helper_img, prompt_style="strict"):
    images = [primary_img, helper_img] if helper_img else [primary_img]
    if prompt_style == "soft":
        return [
            {"role": "system",
             "content": [{"type": "text", "text": _SOFT_SYS_PROMPT}]},
            {"role": "user",
             "content": _build_soft_user_blocks(
                 question["question"], question["answer_choices"], images)},
        ]
    sys_prompt, content = format_spatial_vqa_prompt_policy_plan(
        question=question["question"],
        answer_choices=question["answer_choices"],
        images=images,
    )
    user_blocks = _content_tuples_to_qwen_user_blocks(content)
    return [
        {"role": "system",
         "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": user_blocks},
    ]


# ----------------------------------------------------------------------------
#  Train loop
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    rank, world, local = setup_ddp()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    log(f"world={world} local={local} cuda={torch.cuda.current_device()}", rank)

    # ------------ Wandb (rank 0 only) ------------
    use_wandb = args.wandb_project is not None and is_main(rank)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )
        log(f"wandb: logging to {args.wandb_entity}/{args.wandb_project}", rank)

    # ------------ Trainer (Qwen + LoRA) ------------
    trainer = QwenGRPOTrainer(
        model_name=args.policy_model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Each rank sees only its own GPU (remapped at module top), so the
        # device index inside the rank is always 0.
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        adapter_ckpt=args.resume_adapter,
    )
    if args.resume_adapter and is_main(rank):
        log(f"resumed LoRA adapter from {args.resume_adapter} "
            f"(step counter starts at {args.start_step})", rank)

    # Optimizer over LoRA params only.
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in trainer.model.parameters())
        log(f"trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M", rank)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # ------------ WM + QA rollout ------------
    rollout = WMQARollout(
        model_args=args,
        qa_model_name=args.qa_model_name,
        qa_provider=args.qa_provider,
        work_dir=os.path.join(args.scratch_dir, f"rank{rank}"),
    )

    # ------------ Data ------------
    questions = load_questions(args.train_file, max_images=args.max_images)
    if rank == 0:
        log(f"loaded {len(questions)} train questions", rank)

    # Shard across ranks (round-robin).
    shard = [q for i, q in enumerate(questions) if i % world == rank]
    log(f"local shard size: {len(shard)}", rank)
    rng = random.Random(args.seed + rank)

    # ------------ KL early-stop tracker ------------
    from collections import deque
    kl_window = deque(maxlen=args.kl_early_stop_window)
    kl_aborted = False

    # ------------ Train loop ------------
    metrics_buf = []
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)

    step = args.start_step
    micro_batch_idx = 0
    iter_per_epoch = max(1, len(shard))

    while step < args.max_steps:
        # Sample one question for this micro-batch.
        question = rng.choice(shard)
        primary, helper = prepare_image_pair(question, args.scratch_dir)

        messages = build_policy_messages(question, primary, helper, prompt_style=args.prompt_style)

        # ---- Sample K rollouts ----
        t0 = time.time()
        rollouts = trainer.generate_rollouts(
            messages,
            num_rollouts=args.num_rollouts,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_new_tokens,
        )
        gen_t = time.time() - t0

        # ---- Reward each rollout (parse → WM/QA → reward) ----
        # Deduplicate: rollouts with identical plan share one QA/WM call so
        # reward variance comes from the *policy*, not from GPT-4o noise.
        rewards = []
        per_rollout_logs = []
        n_skip, n_call, n_parse_fail = 0, 0, 0
        n_correct = 0
        plan_cache = {}  # plan_key -> (eval_result, reward, dedup_id)
        next_dedup_id = 0
        created_rollout_tags = []  # for post-backward cleanup
        for j, ro in enumerate(rollouts):
            parsed = _parse_policy_json(
                ro.response_text,
                args.sampling_interval_meter,
                args.sampling_interval_angle,
                max_atomic=args.max_atomic_actions,
            )
            if not parsed["parse_ok"]:
                n_parse_fail += 1
                ro.aux = {"decision": None, "parse_ok": False, "num_atomic": 0}
                rewards.append(args.parse_fail_penalty)
                per_rollout_logs.append({
                    "j": j,
                    "decision": "PARSE_FAIL",
                    "n_actions": 0,
                    "raw_actions": [],
                    "is_correct": False,
                    "qa_response": None,
                    "reward": args.parse_fail_penalty,
                    "plan_text": ro.response_text,
                    "dedup_id": "PF",
                    "cache_hit": False,
                })
                continue

            plan_dict = {
                "decision": parsed["decision"],
                "actions": parsed["atomic_actions"],
                "reason": parsed["reason"],
            }

            # Plan key = decision + canonical atomic action sequence.
            plan_key = (
                parsed["decision"],
                tuple((a["type"], round(float(a["value"]), 4))
                      for a in parsed["atomic_actions"]),
            )

            cache_hit = plan_key in plan_cache
            if cache_hit:
                res, r, dedup_id = plan_cache[plan_key]
            else:
                rollout_tag = (
                    f"mb{micro_batch_idx}_step{step}"
                    f"_q{question['database_idx']}_r{j}"
                )
                created_rollout_tags.append(rollout_tag)
                res = rollout.evaluate_plan(
                    question=question,
                    image_path=primary,
                    helper_image_path=helper,
                    plan=plan_dict,
                    rollout_tag=rollout_tag,
                    cleanup=False,  # cleanup happens after backward (below)
                )
                r = compute_reward(
                    res,
                    action_cost=args.action_cost,
                    parse_fail_penalty=args.parse_fail_penalty,
                    skip_wrong_penalty=args.skip_wrong_penalty,
                )
                dedup_id = next_dedup_id
                next_dedup_id += 1
                plan_cache[plan_key] = (res, r, dedup_id)

            rewards.append(r)
            ro.aux = {
                "decision": res["decision"],
                "is_correct": res["is_correct"],
                "num_atomic": res["num_atomic_actions"],
                "status": res["status"],
                "dedup_id": dedup_id,
                "cache_hit": cache_hit,
            }
            if res["decision"] == "skip":
                n_skip += 1
            elif res["decision"] == "call_wm":
                n_call += 1
            if res["is_correct"]:
                n_correct += 1

            per_rollout_logs.append({
                "j": j,
                "decision": res["decision"],
                "n_actions": res["num_atomic_actions"],
                "raw_actions": parsed["raw_actions"],
                "is_correct": res["is_correct"],
                "qa_response": res["qa_response"],
                "qa_parsed": res["qa_parsed"],
                "reward": r,
                "reason": parsed["reason"],
                "plan_text": ro.response_text,
                "status": res["status"],
                "dedup_id": dedup_id,
                "cache_hit": cache_hit,
            })

        # ---- Group-relative advantages ----
        advantages = compute_group_advantages(rewards)
        for ro, r, a in zip(rollouts, rewards, advantages):
            ro.reward, ro.advantage = r, a

        # ---- Verbose per-rollout dump ----
        if args.verbose_rollouts:
            qa_acc = (n_correct / max(1, len(rollouts)))
            n_unique_plans = len(plan_cache)
            header = (
                f"\n========== rank{rank} step={step} qid={question['database_idx']} "
                f"({question['question_type']}) ==========\n"
                f"Q: {question['question']}\n"
                f"  choices: {question['answer_choices']}\n"
                f"  GT     : {question['correct_answer']}\n"
                f"  ---- summary: K={len(rollouts)} unique_plans={n_unique_plans} "
                f"skip={n_skip} call_wm={n_call} parse_fail={n_parse_fail} "
                f"qa_acc={qa_acc:.2f} "
                f"mean_reward={sum(rewards)/max(1,len(rewards)):.3f} "
                f"gen_t={gen_t:.1f}s ----"
            )
            print(header, flush=True)
            for entry, adv in zip(per_rollout_logs, advantages):
                plan_text = entry["plan_text"].strip().replace("\n", " ")
                if len(plan_text) > args.print_plan_chars:
                    plan_text = plan_text[: args.print_plan_chars] + "...[trunc]"
                if entry["decision"] == "PARSE_FAIL":
                    print(
                        f"  [r{entry['j']:>2}] PARSE_FAIL          "
                        f"reward={entry['reward']:+.3f} adv={adv:+.3f}\n"
                        f"        plan: {plan_text}",
                        flush=True,
                    )
                    continue
                actions_str = ", ".join(
                    f"{a['type']}={a['value']}" for a in entry["raw_actions"]
                ) or "(none)"
                qa_resp = entry["qa_response"] or ""
                qa_resp = qa_resp.strip().replace("\n", " ")
                if len(qa_resp) > args.print_qa_chars:
                    qa_resp = qa_resp[: args.print_qa_chars] + "...[trunc]"
                cache_tag = " (cached)" if entry.get("cache_hit") else ""
                print(
                    f"  [r{entry['j']:>2}] plan#{entry.get('dedup_id', '?')}{cache_tag} "
                    f"{entry['decision']:<8} "
                    f"n_atomic={entry['n_actions']} "
                    f"correct={int(bool(entry['is_correct']))} "
                    f"reward={entry['reward']:+.3f} adv={adv:+.3f} "
                    f"status={entry.get('status', '?')}\n"
                    f"        actions : {actions_str}\n"
                    f"        qa_pick : {entry.get('qa_parsed')!r}\n"
                    f"        qa_resp : {qa_resp}\n"
                    f"        plan    : {plan_text}",
                    flush=True,
                )

        # ---- Early wandb log: rollout / reward metrics ----
        # Logged BEFORE grpo_loss so even if backward crashes the reward and
        # diversity curves still show up on wandb. We use commit=False here
        # and commit=True after the loss step; both write to the same wandb step.
        if use_wandb:
            n_atomic_call = [
                e["n_actions"] for e in per_rollout_logs
                if e["decision"] == "call_wm"
            ]
            early_metrics = {
                "reward/mean":   sum(rewards) / max(1, len(rewards)),
                "reward/min":    min(rewards) if rewards else 0.0,
                "reward/max":    max(rewards) if rewards else 0.0,
                "qa/acc":        n_correct / max(1, len(rollouts)),
                "decision/skip_rate":       n_skip / max(1, len(rollouts)),
                "decision/call_rate":       n_call / max(1, len(rollouts)),
                "decision/parse_fail_rate": n_parse_fail / max(1, len(rollouts)),
                "rollout/mean_n_atomic_when_call":
                    (sum(n_atomic_call) / len(n_atomic_call))
                    if n_atomic_call else 0.0,
                "rollout/n_unique_plans": len(plan_cache),
                "perf/gen_t_sec":  gen_t,
            }
            print(f"[wandb] early-log step={micro_batch_idx} "
                  f"reward_mean={early_metrics['reward/mean']:.3f} "
                  f"qa_acc={early_metrics['qa/acc']:.2f}", flush=True)
            wandb.log(early_metrics, step=micro_batch_idx, commit=False)

        # ---- GRPO loss + backward (per-rollout micro-grad accumulation) ----
        trainer.model.train()
        step_metrics = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "n": 0}
        for ro in rollouts:
            # Skip zero-advantage rollouts for efficiency (their grad is 0 anyway).
            if abs(ro.advantage) < 1e-8:
                continue
            loss, m = trainer.grpo_loss(ro)
            # Scale loss by 1 / (K * grad_accum) so the accumulated grads match
            # taking one step over (K * grad_accum) rollouts.
            loss = loss / (args.num_rollouts * args.grad_accum_steps)
            loss.backward()
            for k in ("loss", "pg_loss", "kl"):
                step_metrics[k] += m[k]
            step_metrics["n"] += 1

        accum_count += 1

        # ---- Cleanup per-rollout scratch dirs now that backward is done ----
        # Disk artifacts (pred.mp4, sampled frames, video.mp4) are no longer
        # needed once backward finished (logprobs use per-question shared
        # images from prepare_image_pair, not these per-rollout dirs).
        # The new/ref logprob recompute already happened inside grpo_loss above,
        # so we can safely rmtree here.
        if not args.keep_rollout_dirs:
            for tag in created_rollout_tags:
                rollout.cleanup_rollout(tag)

        # ---- Optimizer step every grad_accum_steps micro-batches ----
        did_step = False
        if accum_count >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            did_step = True
            step += 1

        # ---- Logging ----
        n_active = step_metrics["n"]
        K = max(1, n_active)
        if n_active == 0:
            warn = "WARN: all_advantages_zero (all K rewards equal -> no learning signal)"
        else:
            warn = None
        info = {
            "qid": question["database_idx"],
            "qtype": question["question_type"],
            "K": args.num_rollouts,
            "n_unique_plans": len(plan_cache),
            "n_active": n_active,
            "n_skip": n_skip,
            "n_call": n_call,
            "n_parse_fail": n_parse_fail,
            "n_correct": n_correct,
            "qa_acc": round(n_correct / max(1, len(rollouts)), 3),
            "rewards": [round(r, 3) for r in rewards],
            "advs": [round(a, 3) for a in advantages],
            "loss": step_metrics["loss"] / K if n_active > 0 else None,
            "pg_loss": step_metrics["pg_loss"] / K if n_active > 0 else None,
            "kl": step_metrics["kl"] / K if n_active > 0 else None,
            "gen_t": round(gen_t, 2),
            "did_step": did_step,
            "step": step,
            "warn": warn,
        }
        if (step % args.log_every) == 0 or did_step:
            log(json.dumps(info), rank)

        # ---- Late wandb log: loss metrics + optim flag (commits this step) ----
        if use_wandb:
            late_metrics = {
                "rollout/n_active":  n_active,
                "train/optim_step":  step,
                "train/did_optim":   int(did_step),
            }
            if n_active > 0:
                # `loss/kl` is the raw K3 estimator. The amount actually
                # added to the optimised loss is kl_beta * that, which is
                # what `loss/kl_weighted` shows — far easier to compare
                # against `loss/pg` to see how much KL is really shaping
                # the gradient.
                kl_raw = step_metrics["kl"] / K
                late_metrics.update({
                    "loss/total":       step_metrics["loss"] / K,
                    "loss/pg":          step_metrics["pg_loss"] / K,
                    "loss/kl":          kl_raw,
                    "loss/kl_weighted": args.kl_beta * kl_raw,
                })
            print(f"[wandb] late-log  step={micro_batch_idx} "
                  f"n_active={n_active} did_step={int(did_step)} "
                  f"pg_loss={late_metrics.get('loss/pg')}", flush=True)
            wandb.log(late_metrics, step=micro_batch_idx, commit=True)
        micro_batch_idx += 1

        # ---- KL safety net ----
        if n_active > 0:
            cur_kl = step_metrics["kl"] / K
            kl_window.append(cur_kl)
            if (len(kl_window) >= args.kl_early_stop_window
                    and sum(kl_window) / len(kl_window)
                        > args.kl_early_stop_threshold):
                rolling = sum(kl_window) / len(kl_window)
                log(
                    f"KL EARLY-STOP: rolling mean over last "
                    f"{args.kl_early_stop_window} micro-batches = "
                    f"{rolling:.3f} > threshold {args.kl_early_stop_threshold}. "
                    "Policy is drifting too far from ref; aborting before collapse.",
                    rank,
                )
                kl_aborted = True
                break

        # ---- Save adapter ----
        if did_step and step > 0 and (step % args.save_every == 0) and is_main(rank):
            ck = os.path.join(args.save_dir, f"adapter_step{step}")
            trainer.save_adapter(ck)
            log(f"saved adapter -> {ck}", rank)

    # ---- Final save ----
    if is_main(rank):
        suffix = "adapter_final_kl_aborted" if kl_aborted else "adapter_final"
        ck = os.path.join(args.save_dir, suffix)
        trainer.save_adapter(ck)
        log(f"final adapter -> {ck}", rank)
        if use_wandb:
            import wandb
            wandb.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
