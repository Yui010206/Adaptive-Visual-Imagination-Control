"""
GRPO trainer for Qwen3-VL-4B with LoRA.

Implements:
  - LoRA-wrapped policy
  - Generation with per-token old logprobs (from generate scores)
  - Recomputation of new/ref logprobs in teacher-forcing mode
  - Group-relative advantages (per-question normalisation)
  - PPO-clip loss + K3 KL penalty to ref policy
  - Single-call `step()` that does loss + backward (caller does optim step)

Design notes:
  * Reference policy = the same model with the LoRA adapter disabled, so we
    don't need to keep a frozen copy in memory.
  * Old logprobs are captured at sampling time from `output_scores`. This is
    the standard PPO practice and avoids a second forward pass.
  * We mask out everything before the first generated token and any padding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor


# --- Workaround: peft 0.19.x + transformers 4.57.x version skew ---
# When DDP is initialised, peft's `set_peft_model_state_dict` calls
# `_maybe_shard_state_dict_for_tp`, which unconditionally does
# `from transformers.integrations.tensor_parallel import EmbeddingParallel`
# at function entry. EmbeddingParallel was added in a transformers version
# newer than 4.57, so the import raises ImportError before peft can even
# check whether the model uses TP. For DDP-only training (no `_hf_tp_plan`
# on any module) the function is a strict no-op anyway, so we short-circuit.
try:
    import peft.utils.save_and_load as _peft_sl

    _orig_maybe_shard = _peft_sl._maybe_shard_state_dict_for_tp

    def _maybe_shard_state_dict_for_tp_safe(model, state_dict, adapter_name):
        has_tp = any(getattr(m, "_hf_tp_plan", None) is not None
                     for m in model.modules())
        if not has_tp:
            return  # DDP/single-GPU: nothing to shard.
        return _orig_maybe_shard(model, state_dict, adapter_name)

    _peft_sl._maybe_shard_state_dict_for_tp = _maybe_shard_state_dict_for_tp_safe
except Exception:
    # Older peft without this helper — nothing to patch.
    pass


@dataclass
class Rollout:
    """One sampled rollout for a single (question, image) prompt."""
    messages: list                  # the system+user messages used for sampling
    response_text: str              # decoded response
    response_token_ids: torch.Tensor  # (T_resp,)  generated tokens (no prompt)
    old_logprobs: torch.Tensor      # (T_resp,)   logprob of each gen token at sample time
    reward: float = 0.0             # set by the rollout/reward function
    advantage: float = 0.0          # set after group normalisation
    aux: dict = field(default_factory=dict)  # for logging (decision, num_actions, etc.)


def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (B, T, V); labels: (B, T). Return per-token log p(label | ...)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


class QwenGRPOTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        device: str = "cuda",
        attn_implementation: Optional[str] = None,
        dtype: str = "bfloat16",
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        adapter_ckpt: Optional[str] = None,
    ):
        self.device = device
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(dtype, torch.bfloat16)

        load_kwargs = {"dtype": torch_dtype, "trust_remote_code": True}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        base = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        base.to(device)

        if adapter_ckpt:
            self.model = PeftModel.from_pretrained(base, adapter_ckpt, is_trainable=True)
        else:
            target_modules = lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.model = get_peft_model(base, lora_cfg)

        # gradient_checkpointing recomputes the forward, but since LoRA params
        # are deep inside the network, the inputs_embeds at the recomputed
        # forward have requires_grad=False -> the recomputed forward yields a
        # tensor without grad_fn, so loss.backward() raises "element 0 of
        # tensors does not require grad". Forcing the input embeddings to
        # require grad re-enables the autograd graph through checkpointing.
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # Special multimodal placeholder token IDs. With T=1.2 + top_p=1.0 +
        # top_k=0 the rollout sometimes samples one of these into the
        # response (~1-in-thousands), which then breaks teacher-forcing in
        # compute_logprobs because the count of image-tokens in input_ids
        # no longer matches the count of image features from the vision
        # encoder ("tokens: 325, features 324"). We forbid them at sampling
        # time and also clamp them out defensively when re-feeding the
        # response, so a stale rollout from before this fix can't crash us.
        base_cfg = getattr(self.model, "config", None)
        # PeftModel's `.config` is its own; the real model lives one level deeper.
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            base_cfg = self.model.base_model.model.config
        self._special_visual_token_ids: List[int] = sorted({
            t for t in (
                getattr(base_cfg, "image_token_id", None),
                getattr(base_cfg, "video_token_id", None),
                getattr(base_cfg, "vision_start_token_id", None),
                getattr(base_cfg, "vision_end_token_id", None),
            )
            if isinstance(t, int)
        })

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_rollouts(
        self,
        messages: list,
        num_rollouts: int,
        temperature: float = 1.2,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 512,
    ) -> List[Rollout]:
        """Sample K rollouts for the same prompt. Returns the response tokens
        and per-token sampling-time logprobs for each rollout."""
        self.model.eval()

        # Build inputs once; expand to K rollouts via num_return_sequences.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = dict(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_rollouts,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if self._special_visual_token_ids:
            # Setting these logits to -inf at every step prevents the model
            # from ever placing image/video placeholder tokens inside its
            # generated text. Cheaper than bad_words_ids and doesn't need
            # any tokeniser-side processing.
            gen_kwargs["suppress_tokens"] = self._special_visual_token_ids
        out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        seqs = out.sequences  # (K, prompt_len + T_resp_max)
        gen_ids = seqs[:, prompt_len:]  # (K, T_resp_max)

        # output_scores is a tuple of T_resp_max tensors of shape (K, V),
        # the *unnormalized* logits at each generation step. The naive
        # `stack -> log_softmax -> gather` materialises a (T, K, V) fp32
        # tensor; for soft-prompt rollouts that emit ~500 tokens with
        # K=16 and V≈152k that's ~5 GB and OOMs. We only need the
        # logprob of the actually-sampled token per step, so do
        # log_softmax + gather one step at a time and free as we go.
        T_resp = len(out.scores)
        gathered_steps = []
        for t in range(T_resp):
            score_t = out.scores[t]                         # (K, V)
            lp_t = F.log_softmax(score_t.float(), dim=-1)   # (K, V) on-GPU temp
            gathered_steps.append(
                lp_t.gather(-1, gen_ids[:, t:t + 1]).squeeze(-1)  # (K,)
            )
            del lp_t
        gathered = torch.stack(gathered_steps, dim=1)       # (K, T_resp)
        del gathered_steps

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        rollouts: List[Rollout] = []
        for k in range(num_rollouts):
            tok = gen_ids[k]
            # cut at first eos (inclusive) so the trailing tokens don't pollute logprobs
            eos_pos = (tok == eos_id).nonzero(as_tuple=False)
            if len(eos_pos) > 0:
                cut = int(eos_pos[0].item()) + 1
                tok = tok[:cut]
                lp = gathered[k, :cut]
            else:
                # also drop pad tokens at the tail (in case the model never emitted EOS)
                non_pad = (tok != pad_id).nonzero(as_tuple=False)
                cut = int(non_pad[-1].item()) + 1 if len(non_pad) > 0 else tok.shape[0]
                tok = tok[:cut]
                lp = gathered[k, :cut]

            text = self.tokenizer.decode(tok, skip_special_tokens=True)
            rollouts.append(Rollout(
                messages=messages,
                response_text=text,
                response_token_ids=tok.detach().cpu(),
                old_logprobs=lp.detach().cpu(),
            ))

        return rollouts

    # ------------------------------------------------------------------
    #  Logprob recomputation (teacher forcing)
    # ------------------------------------------------------------------
    def _build_tf_inputs(self, rollout: Rollout):
        """Concat prompt + response and return the inputs needed for a forward
        pass + the response token mask (1 over response positions, else 0)."""
        prompt_inputs = self.processor.apply_chat_template(
            rollout.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        prompt_ids = prompt_inputs["input_ids"]      # (1, P)
        attn_mask = prompt_inputs["attention_mask"]  # (1, P)

        resp_ids = rollout.response_token_ids.to(self.device).unsqueeze(0)  # (1, T)
        # Defensive sanitise: if a stale rollout (sampled before suppress_tokens
        # was added, or sampled by another implementation) contains an
        # image/video placeholder token, replace it with eos. Otherwise
        # `get_placeholder_mask` counts it as a real image slot and crashes
        # ("Image features and image tokens do not match") because no extra
        # vision feature was produced for it.
        if self._special_visual_token_ids:
            eos = self.tokenizer.eos_token_id
            mask = torch.zeros_like(resp_ids, dtype=torch.bool)
            for t in self._special_visual_token_ids:
                mask |= (resp_ids == t)
            if mask.any():
                resp_ids = resp_ids.clone()
                resp_ids[mask] = eos
        full_ids = torch.cat([prompt_ids, resp_ids], dim=1)                  # (1, P+T)
        full_attn = torch.cat([attn_mask, torch.ones_like(resp_ids)], dim=1)

        # Build the multimodal kwargs. Anything that's a per-token tensor
        # aligned with input_ids (shape (B, P)) needs to be extended with T
        # text-token zeros so it matches the concatenated full_ids. Examples:
        #   - mm_token_type_ids (Qwen3-VL) — modality marker
        #   - token_type_ids    (BERT-style)
        # Non-aligned tensors (pixel_values, image_grid_thw, ...) are passed
        # through as-is.
        extra = {}
        P = prompt_ids.shape[1]
        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == P \
                    and v.shape[0] == prompt_ids.shape[0]:
                pad = torch.zeros_like(resp_ids, dtype=v.dtype)
                extra[k] = torch.cat([v, pad], dim=1)
            else:
                extra[k] = v

        return full_ids, full_attn, extra, prompt_ids.shape[1]

    def compute_logprobs(self, rollout: Rollout, use_adapter: bool):
        """Return per-token logprob of the response tokens under the model.
        With `use_adapter=False`, the LoRA adapter is disabled to give π_ref."""
        full_ids, full_attn, extra, prompt_len = self._build_tf_inputs(rollout)

        ctx = self.model.disable_adapter() if not use_adapter else _NullCtx()
        with ctx:
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_attn,
                **extra,
                use_cache=False,
            )
        logits = outputs.logits  # (1, P+T, V)

        # logits[t] predicts token at t+1, so for response positions
        # [P, P+1, ..., P+T-1] the predicting logits live at
        # [P-1, P, ..., P+T-2].
        resp_len = full_ids.shape[1] - prompt_len
        pred_logits = logits[:, prompt_len - 1: prompt_len - 1 + resp_len, :]  # (1, T, V)
        labels = full_ids[:, prompt_len:prompt_len + resp_len]                 # (1, T)
        return _gather_logprobs(pred_logits, labels).squeeze(0)                # (T,)

    # ------------------------------------------------------------------
    #  GRPO loss
    # ------------------------------------------------------------------
    def grpo_loss(self, rollout: Rollout):
        """Return a scalar loss (and a dict of metrics) for one rollout.
        Caller is responsible for backward + optimizer.step()."""
        adv = float(rollout.advantage)
        old_lp = rollout.old_logprobs.to(self.device)

        new_lp = self.compute_logprobs(rollout, use_adapter=True)
        with torch.no_grad():
            ref_lp = self.compute_logprobs(rollout, use_adapter=False)

        # Length-mismatch shouldn't happen but guard anyway.
        T = min(new_lp.shape[0], old_lp.shape[0], ref_lp.shape[0])
        new_lp, old_lp, ref_lp = new_lp[:T], old_lp[:T], ref_lp[:T]

        # PPO-style clipped objective
        ratio = (new_lp - old_lp).exp()
        unclipped = ratio * adv
        clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
        pg_per_token = -torch.min(unclipped, clipped)
        pg_loss = pg_per_token.mean()

        # K3 KL estimator: exp(ref - new) - (ref - new) - 1  (always >= 0)
        log_ratio_ref = ref_lp - new_lp
        kl_per_token = log_ratio_ref.exp() - log_ratio_ref - 1
        kl_loss = kl_per_token.mean()

        loss = pg_loss + self.kl_beta * kl_loss

        with torch.no_grad():
            metrics = {
                "loss": float(loss.detach()),
                "pg_loss": float(pg_loss.detach()),
                "kl": float(kl_loss.detach()),
                "advantage": adv,
                "ratio_mean": float(ratio.mean().detach()),
                "resp_len": int(T),
            }
        return loss, metrics

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    def save_adapter(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ----- Group-relative advantage helper -----
def compute_group_advantages(rewards: List[float], eps: float = 1e-6) -> List[float]:
    """Standard GRPO group normalisation: a_i = (r_i - mean) / (std + eps)."""
    if not rewards:
        return []
    t = torch.tensor(rewards, dtype=torch.float32)
    mean = t.mean()
    std = t.std(unbiased=False)
    return ((t - mean) / (std + eps)).tolist()
