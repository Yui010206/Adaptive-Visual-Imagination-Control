"""
Qwen3-VL policy model.

A standalone wrapper that turns Qwen3-VL into a POLICY model that does
gating + action planning for the spatial-VQA pipeline. It reuses the
existing `format_spatial_vqa_prompt_policy_plan` (the v4 prompt with
`decision` ∈ {skip, call_wm}) so the JSON schema matches what the rest
of the codebase expects.

QA usage is intentionally not handled here.
"""

import json
import os
import re
import sys

# Allow `python utils/qwen_policy.py` from the repo root by ensuring the repo
# root is on sys.path before the `utils.*` import below.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.prompt_formatting import format_spatial_vqa_prompt_policy_plan


_ACTION_TYPES = ("move-forward", "turn-left", "turn-right")


# Permissive parser for free-form action strings the model sometimes emits
# inside `actions`, e.g. "turn-right 9 degrees", "move forward 0.5 m",
# "turn left by 18°". Captures (verb, magnitude). The verb is then
# normalised to one of `_ACTION_TYPES` (lowercase, hyphenated).
_ACTION_STRING_RE = re.compile(
    r"(move[-\s]*forward|turn[-\s]*left|turn[-\s]*right)"
    r"\s*(?:by\s+)?"
    r"(\d+(?:\.\d+)?)"
    r"\s*(?:degrees|degree|deg|°|m|meters|meter)?",
    re.IGNORECASE,
)


def _normalise_action(a):
    """Coerce one action item into a canonical {"type", "value"} dict, or
    return None if it's unrecoverable. Accepts:
      * dict already in the right shape ({"type": <action>, "value": <num>})
      * dict with the action name AS the key, e.g. {"turn-left": 18}
      * free-form string, e.g. "turn-right 9 degrees" / "move forward 0.5 m"
    """
    if isinstance(a, dict):
        # Canonical form
        raw_type = a.get("type")
        raw_value = a.get("value")
        if isinstance(raw_type, str):
            t = raw_type.strip().lower().replace(" ", "-")
            try:
                v = float(raw_value)
            except (TypeError, ValueError):
                return None
            if t in _ACTION_TYPES and v > 0:
                return {"type": t, "value": v}
            return None
        # Sometimes models emit {"turn-left": 18} with no "type" key.
        for k, v in a.items():
            if not isinstance(k, str):
                continue
            t = k.strip().lower().replace(" ", "-")
            if t in _ACTION_TYPES:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return None
                if fv > 0:
                    return {"type": t, "value": fv}
        return None
    if isinstance(a, str):
        m = _ACTION_STRING_RE.search(a)
        if not m:
            return None
        t = m.group(1).strip().lower().replace(" ", "-")
        if t.startswith("move"):
            t = "move-forward"
        elif t.startswith("turn-left") or t.startswith("turn left"):
            t = "turn-left"
        elif t.startswith("turn-right") or t.startswith("turn right"):
            t = "turn-right"
        if t not in _ACTION_TYPES:
            return None
        try:
            v = float(m.group(2))
        except ValueError:
            return None
        if v <= 0:
            return None
        return {"type": t, "value": v}
    return None


def _salvage_policy_object(obj):
    """Coerce a partially-broken policy JSON into a canonical
    {"decision", "actions"} dict, or None if it's unrecoverable.

    Handles two common high-T failure modes that the strict parser
    would have rejected:

      * `decision` set to an action name (e.g. "turn-left") instead
        of "skip" / "call_wm". We infer: any non-empty `actions` list
        means the model intended to call the WM; an empty list is a
        skip.
      * `actions` items that are bare strings ("turn-right 9 degrees")
        instead of {"type", "value"} dicts. _normalise_action parses
        them with a permissive regex.

    Salvage is purely a signal-quality lift: a recoverable rollout that
    was hitting parse_fail_penalty=-0.5 now lands at the (typically
    milder) call_wm-wrong reward, so GRPO can distinguish "format
    confused" from "no JSON at all".
    """
    if not isinstance(obj, dict):
        return None

    # ---- Normalise actions ----
    raw_actions = obj.get("actions", [])
    if not isinstance(raw_actions, list):
        return None
    cleaned: list = []
    for a in raw_actions:
        na = _normalise_action(a)
        if na is None:
            # Even one bad action invalidates the plan — we can't safely
            # guess what the model meant for the remainder.
            return None
        cleaned.append(na)

    # ---- Normalise decision ----
    raw_decision = obj.get("decision")
    decision = None
    if isinstance(raw_decision, str):
        d = raw_decision.strip().lower()
        if d in ("skip", "call_wm"):
            decision = d
        # Common synonyms the model invents under high T.
        elif d in ("call", "call-wm", "callwm", "explore", "explore_first",
                   "use_wm", "use-wm", "world_model"):
            decision = "call_wm"
        elif d in ("answer", "answer_now", "direct", "no", "stop", "done"):
            decision = "skip"
    if decision is None:
        # Fall back: infer from the actions list.
        decision = "call_wm" if cleaned else "skip"

    # ---- Consistency: enforce the mutual contract ----
    if decision == "skip" and cleaned:
        # `skip` with non-empty actions is contradictory; trust the
        # actions and call WM.
        decision = "call_wm"
    if decision == "call_wm" and not cleaned:
        # `call_wm` with no actions has nothing to feed to the WM.
        return None

    return {"decision": decision, "actions": cleaned}


def _extract_last_json_object(text: str):
    """Walk the text right-to-left, find each `}`, locate its matching `{`
    via a balance counter, and try to json.loads the substring. Returns
    the first dict that parses AND contains the `decision` key, or None.

    Robust to:
      * reasoning text containing stray `{` (soft-prompt rollouts often
        echo the schema template or use prose like "{object}");
      * markdown code fences (the ```json...``` markers are outside the
        balanced braces so they're ignored);
      * trailing chatter after the JSON (we walk from the right so a
        closing brace inside an inner dict that's earlier in the string
        won't shadow the real outer one);
      * nested objects in `actions: [{...}, {...}]`.
    """
    if not text:
        return None
    n = len(text)
    # Iterate over closing braces from right to left.
    i = n - 1
    while i >= 0:
        if text[i] != "}":
            i -= 1
            continue
        # Walk left, balancing braces, to find the matching `{`.
        depth = 1
        j = i - 1
        while j >= 0:
            c = text[j]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    blob = text[j:i + 1]
                    try:
                        obj = json.loads(blob)
                    except Exception:
                        obj = None
                    # A policy JSON must at minimum have *some* policy
                    # field — `decision` (canonical), or `actions`
                    # (model wrote actions but forgot the decision key,
                    # which the salvage path can still recover).
                    if isinstance(obj, dict) and ("decision" in obj or "actions" in obj):
                        return obj
                    # Not a valid policy JSON; resume scanning further left
                    # by skipping past this `}` and looking for an earlier one.
                    break
            j -= 1
        i -= 1
    return None


def _content_tuples_to_qwen_user_blocks(content):
    """
    Convert the (text,) / (text, image_path) tuples produced by
    `format_spatial_vqa_prompt_policy_plan` into Qwen3-VL `user` content blocks.

    Each (text, img) tuple becomes [text-block, image-block] in order, so the
    image is anchored right after its label text — same layout as the GPT path.
    """
    blocks = []
    for c in content:
        if not c:
            continue
        text = c[0] if len(c) >= 1 else ""
        if text:
            blocks.append({"type": "text", "text": text})
        if len(c) == 2:
            blocks.append({"type": "image", "image": c[1]})
    return blocks


def _parse_policy_json(text, interval_meter, interval_angle, max_atomic=None):
    """
    Parse the policy_plan JSON (gating + planning) and expand into atomic steps.

    Returns:
        {
          "decision": "skip" | "call_wm" | None,
          "reason": str,
          "raw_actions": [{"type": str, "value": float}, ...],
          "atomic_actions": [{"type": str, "value": float}, ...],
          "parse_ok": bool,
        }
    """
    out = {
        "decision": None,
        "reason": "",
        "raw_actions": [],
        "atomic_actions": [],
        "parse_ok": False,
    }
    if not text:
        return out
    obj = _extract_last_json_object(text)
    if obj is None:
        return out

    # Run the JSON object through the salvage path. It handles strict
    # canonical inputs as a no-op and rescues the two common high-T
    # failure modes (decision is an action name; actions are bare
    # strings instead of dicts). If salvage returns None the JSON is
    # genuinely unrecoverable and we fall through to parse_fail.
    normalised = _salvage_policy_object(obj)
    if normalised is None:
        return out

    decision = normalised["decision"]
    out["decision"] = decision
    raw_reason = obj.get("reason")
    out["reason"] = raw_reason.strip() if isinstance(raw_reason, str) else ""

    if decision == "skip":
        out["parse_ok"] = True
        return out

    # call_wm: expand each (already-validated) raw action into atomic steps.
    raw, atomic = [], []
    for a in normalised["actions"]:
        t = a["type"]
        v = a["value"]
        raw.append({"type": t, "value": v})

        step = interval_meter if t == "move-forward" else interval_angle
        n_steps = max(1, int(round(v / step))) if step > 0 else 1
        for _ in range(n_steps):
            atomic.append({"type": t, "value": step})
            if max_atomic is not None and len(atomic) >= max_atomic:
                break
        if max_atomic is not None and len(atomic) >= max_atomic:
            break

    out["raw_actions"] = raw
    out["atomic_actions"] = atomic
    out["parse_ok"] = True
    return out


class Qwen3VLPolicy:
    """
    A POLICY model wrapper around Qwen3-VL-{4B,8B}-Instruct.

    Given (question, answer_choices, images) it produces:
      - decision  ∈ {"skip", "call_wm"}
      - reason
      - raw / atomic action plans
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        dtype="auto",
        device_map="auto",
        attn_implementation=None,
        interval_meter=0.25,
        interval_angle=9,
        max_atomic_actions=6,
        lora_ckpt=None,
    ):
        kwargs = {"dtype": dtype, "device_map": device_map}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        base = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)

        if lora_ckpt:
            from peft import PeftModel
            print(f"[Qwen3VLPolicy] applying LoRA adapter from {lora_ckpt}",
                  flush=True)
            self.model = PeftModel.from_pretrained(base, lora_ckpt,
                                                   is_trainable=False)
            # Merge so generate() doesn't pay the adapter overhead at inference.
            try:
                self.model = self.model.merge_and_unload()
                print("[Qwen3VLPolicy] adapter merged into base weights.",
                      flush=True)
            except Exception as e:
                # Some PeftModel configs don't support merge; running with the
                # adapter wrapped is also fine.
                print(f"[Qwen3VLPolicy] merge_and_unload failed ({e}); "
                      "running with PeftModel wrapper.", flush=True)
        else:
            self.model = base

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.lora_ckpt = lora_ckpt
        self.interval_meter = interval_meter
        self.interval_angle = interval_angle
        self.max_atomic_actions = max_atomic_actions

    def _build_messages(self, sys_prompt, content):
        user_blocks = _content_tuples_to_qwen_user_blocks(content)
        messages = []
        if sys_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            })
        messages.append({"role": "user", "content": user_blocks})
        return messages

    @torch.no_grad()
    def _generate(self, messages, max_new_tokens=512, temperature=0.0, top_p=1.0):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text

    def plan(
        self,
        question,
        answer_choices,
        images,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        max_retries=3,
    ):
        """
        Run the gating-policy on a single (question, images) instance.

        Args:
            question (str): the spatial-VQA question.
            answer_choices (list[str]).
            images (list[str]): local image paths (or URLs supported by Qwen3-VL).
            max_new_tokens (int).
            temperature (float): 0 = greedy.
            top_p (float).
            max_retries (int): re-decode if JSON parse fails.

        Returns:
            dict: {
                "raw_response": str,
                "decision": str | None,
                "reason": str,
                "raw_actions": list,
                "atomic_actions": list,
                "parse_ok": bool,
            }
        """
        sys_prompt, content = format_spatial_vqa_prompt_policy_plan(
            question=question,
            answer_choices=answer_choices,
            images=images,
        )
        messages = self._build_messages(sys_prompt, content)

        raw, parsed = "", {"parse_ok": False}
        for _ in range(max(1, max_retries)):
            raw = self._generate(messages, max_new_tokens, temperature, top_p)
            parsed = _parse_policy_json(
                raw,
                self.interval_meter,
                self.interval_angle,
                max_atomic=self.max_atomic_actions,
            )
            if parsed["parse_ok"]:
                break

        return {
            "raw_response": raw,
            "decision": parsed["decision"],
            "reason": parsed["reason"],
            "raw_actions": parsed["raw_actions"],
            "atomic_actions": parsed["atomic_actions"],
            "parse_ok": parsed["parse_ok"],
        }


if __name__ == "__main__":
    # Tiny smoke test using the first question of data/val.json.
    import argparse
    import json as _json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--input_file",
        default="data/test.json",
    )
    parser.add_argument("--num_questions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        questions = _json.load(f)

    policy = Qwen3VLPolicy(model_name=args.model_name)

    for q in questions[: args.num_questions]:
        print(f"\n===== QID {q['database_idx']} type={q['question_type']} =====")
        print("Q:", q["question"])
        result = policy.plan(
            question=q["question"],
            answer_choices=q["answer_choices"],
            images=q["img_paths"],
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        print("decision:", result["decision"])
        print("reason  :", result["reason"])
        print("raw_actions:", result["raw_actions"])
        print("atomic_actions:", result["atomic_actions"])
        if not result["parse_ok"]:
            print("[!] parse failed — raw response:")
            print(result["raw_response"])
