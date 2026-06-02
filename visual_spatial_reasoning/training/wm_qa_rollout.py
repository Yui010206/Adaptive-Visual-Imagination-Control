"""
World-model + GPT-4o QA rollout.

Wraps the heavy WM execution and QA prompting from `pipeline_avic.py`
so that the GRPO trainer can call `evaluate_plan(question, image, plan_dict)`
and get back a reward in one call. Reuses the existing helper methods on
`SpatialVQAPipelineSVC` via subclassing — we only override `__init__` to skip
the question loading and policy-model loading we don't need at training time.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

import cv2

# Pipeline imports — same conventions as pipeline_avic.py runs.
from pipelines.pipeline_avic import (
    SpatialVQAPipelineSVC,
    resize_to_short_side,
)
from stable_virtual_camera.demo import Model
from utils.vlm_wrapper import VLMWrapper


def _parse_qa_choice(response: str, answer_choices):
    if not response:
        return None
    resp_l = response.lower()
    last_line = resp_l.split("\n")[-1] if "\n" in resp_l else resp_l
    for c in answer_choices:
        if c.lower() in last_line:
            return c
    for c in answer_choices:
        if c.lower() in resp_l:
            return c
    return None


class WMQARollout(SpatialVQAPipelineSVC):
    """A drop-in helper that owns the WM + QA model and exposes a single
    `evaluate_plan(...)` entry point. Inherits the WM helpers from the
    full pipeline class so we don't duplicate `_simulate_one_sequence` etc.
    """

    # Override to skip everything we don't need (questions, policy model,
    # results dict, output dir). We only need: model_args, vlm (for QA),
    # and global_model (the WM).
    def __init__(self, model_args, qa_model_name="gpt-4o", qa_provider="azure",
                 work_dir="/tmp/grpo_rollouts"):
        # Do NOT call super().__init__() — that would try to parse args, load
        # questions, and load a Qwen policy model. We're at training time.
        self.model_args = model_args

        # QA model (gpt-4o by default) — used for `answer_baseline` and
        # `answer_scaling_no_cot` prompts.
        self.vlm = VLMWrapper(
            model_name=qa_model_name,
            qa_model_name=None,
            provider=qa_provider,
        )
        self.qa_model = None  # we only have one model on self.vlm

        # Heavy WM model.
        self.global_model = Model()

        # No on-disk policy.
        self.qwen_policy = None

        # Where to dump per-rollout artifacts (videos, frames). The training
        # loop typically passes cleanup=False to evaluate_plan and calls
        # cleanup_rollout() explicitly after backward, so the dirs stay alive
        # through QA + new/ref logprob recomputation.
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def cleanup_rollout(self, rollout_tag: str):
        """Remove one rollout's scratch dir. Trainer calls this after backward
        to free disk while still allowing inspection during the rollout."""
        rollout_dir = os.path.join(self.work_dir, rollout_tag)
        if os.path.isdir(rollout_dir):
            shutil.rmtree(rollout_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    #  Public entry-point
    # ------------------------------------------------------------------
    def evaluate_plan(
        self,
        question: dict,
        image_path: str,
        helper_image_path: Optional[str],
        plan: dict,
        rollout_tag: str,
        cleanup: bool = True,
    ):
        """
        Args:
            question: dict with keys {question, answer_choices, correct_answer}.
            image_path: path to the (already resized) primary image on disk.
            helper_image_path: optional second image; pass None if not used.
            plan: parsed policy output: {"decision": "skip"|"call_wm",
                                        "actions": [{type, value}, ...],
                                        "reason": str}.
            rollout_tag: short string used in scratch-dir naming.

        Returns dict with:
            qa_response, qa_parsed, is_correct, decision,
            num_atomic_actions, status.
        """
        decision = (plan or {}).get("decision", "skip")
        atomic_actions = (plan or {}).get("actions", []) if decision == "call_wm" else []

        rollout_dir = os.path.join(self.work_dir, rollout_tag)
        os.makedirs(rollout_dir, exist_ok=True)

        # svc_main hard-codes the scene path as
        #   <save_dir>/step_0/img_0.png
        # (see stable_virtual_camera/demo.py:370). Stage symlinks so that
        # rule resolves to the cached resized images we already have.
        step0_dir = os.path.join(rollout_dir, "step_0")
        os.makedirs(step0_dir, exist_ok=True)
        staged_primary = os.path.join(step0_dir, "img_0.png")
        if not os.path.exists(staged_primary):
            try:
                os.symlink(os.path.abspath(image_path), staged_primary)
            except OSError:
                # fallback: fs that doesn't support symlinks (rare on /tmp)
                shutil.copyfile(image_path, staged_primary)
        staged_helper = None
        if helper_image_path:
            staged_helper = os.path.join(step0_dir, "helper_img.png")
            if not os.path.exists(staged_helper):
                try:
                    os.symlink(os.path.abspath(helper_image_path), staged_helper)
                except OSError:
                    shutil.copyfile(helper_image_path, staged_helper)

        # default fallbacks
        qa_response, qa_parsed, is_correct = None, None, False
        status = "ok"

        try:
            if decision == "call_wm" and atomic_actions:
                action_ids, action_seq_strs = self._plan_to_action_ids(atomic_actions)
                cap = self.model_args.max_action_ids_cap
                action_ids = action_ids[:cap]
                action_seq_strs = action_seq_strs[:cap]

                if not action_ids:
                    status = "empty_after_cap"
                    decision = "skip"  # treat as skip
                else:
                    saved, ordered, folder_name = self._simulate_one_sequence(
                        image_path=staged_primary,
                        step_idx=0,
                        action_ids=action_ids,
                        action_seq_strs=action_seq_strs,
                        model_args=self.model_args,
                        save_dir=rollout_dir,
                        sampling_interval_angle=self.model_args.sampling_interval_angle,
                        sampling_interval_meter=self.model_args.sampling_interval_meter,
                    )
                    
                    print('order'   , ordered)

                    if not ordered:
                        status = "wm_no_frames"
                        decision = "skip"
                    else:
                        sys_prompt, content = self.vlm.format_prompt(
                            prompt_type="answer_scaling_no_cot",
                            question=question["question"],
                            answer_choices=question["answer_choices"],
                            images=[image_path, helper_image_path]
                                   if helper_image_path else [image_path],
                            action_consequences=list(ordered),
                        )
                        qa_response = self.vlm.run_prompt(
                            "answer_scaling_no_cot", sys_prompt, content
                        )

            if decision == "skip" or qa_response is None:
                # baseline QA path
                sys_prompt, content = self.vlm.format_prompt(
                    prompt_type="answer_baseline",
                    question=question["question"],
                    answer_choices=question["answer_choices"],
                    images=[image_path, helper_image_path]
                           if helper_image_path else [image_path],
                )
                qa_response = self.vlm.run_prompt("answer_baseline", sys_prompt, content)

            qa_parsed = _parse_qa_choice(qa_response, question["answer_choices"])
            is_correct = (
                qa_parsed is not None
                and qa_parsed.lower() == question["correct_answer"].lower()
            )

        except Exception as e:
            import traceback
            status = f"exception: {type(e).__name__}: {e}"
            print(
                f"[WMQARollout] {rollout_tag} -> EXCEPTION: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()

        finally:
            if cleanup and os.path.isdir(rollout_dir):
                shutil.rmtree(rollout_dir, ignore_errors=True)

        # Final empty-response sanity warning so silent failures aren't silent.
        if qa_response is None or qa_response == "":
            print(
                f"[WMQARollout] {rollout_tag} -> empty QA response "
                f"(decision={decision}, status={status})",
                flush=True,
            )

        return {
            "qa_response": qa_response,
            "qa_parsed": qa_parsed,
            "is_correct": is_correct,
            "decision": decision,
            "num_atomic_actions": len(atomic_actions),
            "status": status,
        }


def compute_reward(
    eval_result: dict,
    action_cost: float = 0.1,
    parse_fail_penalty: float = -0.5,
    skip_wrong_penalty: float = 0.0,
):
    """
    reward = +1 (correct) or 0 (wrong)   - action_cost * num_atomic_actions

    skip path uses num_atomic_actions = 0 so by default its wrong-reward
    is 0.0, which is *better* than call_wm+wrong (= -action_cost * n).
    That bias is half the reason policies collapse to "always skip": when
    the model is uncertain, skipping-and-being-wrong is rewarded at
    least as well as taking a chance with the world model.

    `skip_wrong_penalty` (>=0) subtracts an extra penalty when the
    decision was `skip` AND the answer was wrong, so:
        skip + correct       :  +1.0
        skip + wrong         :   0.0 - skip_wrong_penalty
        call_wm + correct    :  +1.0 - action_cost * n
        call_wm + wrong      :   0.0 - action_cost * n
    Set skip_wrong_penalty > action_cost * (typical n) to make the model
    prefer trying WM over an uninformed skip.

    A parse-fail (signalled upstream by passing an "invalid" plan)
    returns `parse_fail_penalty`; the caller is responsible for that.
    """
    if eval_result is None:
        return parse_fail_penalty
    if eval_result["status"].startswith("exception"):
        return parse_fail_penalty

    base = 1.0 if eval_result["is_correct"] else 0.0
    cost = action_cost * eval_result["num_atomic_actions"]
    reward = base - cost
    if (not eval_result["is_correct"]) and eval_result.get("decision") == "skip":
        reward -= skip_wrong_penalty
    return reward


def prepare_image_pair(question, scratch_root, target_short=512):
    """Replicates the resize-to-512-short-side preprocessing the original
    pipeline does on raw question images. Returns (primary_path, helper_path|None).
    """
    qid = question["database_idx"]
    qdir = os.path.join(scratch_root, f"q_{qid}_imgs")
    os.makedirs(qdir, exist_ok=True)

    paths = []
    for i, src in enumerate(question["img_paths"]):
        dst = os.path.join(qdir, f"img_{i}.png")
        if not os.path.exists(dst):
            img = cv2.imread(src)
            img = resize_to_short_side(img, target_short=target_short)
            cv2.imwrite(dst, img)
        paths.append(dst)

    primary = paths[0]
    helper = paths[1] if len(paths) > 1 else None
    return primary, helper
