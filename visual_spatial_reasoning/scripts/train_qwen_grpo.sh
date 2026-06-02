#!/usr/bin/env bash
# 8-GPU torchrun launcher for online GRPO training of the Qwen2.5-VL-7B policy.
# Effective batch = 8 GPUs * per_device_batch_size(=1) * grad_accum_steps(=1) = 8 questions/step.
# (Reverted to the step-140 config; the step-140 adapter beat base by +6 pts overall.)

export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./

# ---- Azure GPT-4o (used by the QA model in the rollout) ----
# Set these in your shell/env before launching; do NOT hardcode real keys here.
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-YOUR_AZURE_OPENAI_API_KEY}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://YOUR_RESOURCE.cognitiveservices.azure.com/}"

# ---- Wandb (optional; leave WANDB_API_KEY unset to disable logging) ----
export WANDB_API_KEY="${WANDB_API_KEY:-}"
wandb_project="avic"
wandb_entity=""   # your wandb entity/username, or leave empty
wandb_run_name="grpo_qwen25vl_7b_soft_resume140_$(date +%Y%m%d_%H%M%S)"

# Reduce CUDA fragmentation under heavy alloc pressure.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Switched from Qwen3-VL-4B (entropy collapses on JSON schema) to
# Qwen2.5-VL-7B per probe results: T=1.0 + top_p=0.95 yields ~4 unique
# plans per K=8 group on real training data.
policy_model_name="Qwen/Qwen2.5-VL-7B-Instruct"

train_file="data/train_balanced_grpo.json"  # GPT-4o-prescored 30 easy_skip / 70 needs_wm mix
save_dir="nips_results/grpo_qwen25vl_7b_8gpu"  # soft-prompt run; new dir so strict checkpoints aren't clobbered
scratch_dir="tmp/grpo_scratch_$USER"

# ---- GRPO hyperparams ----
num_rollouts=16                         # K rollouts per question
# Soft-prompt rollout sampling. T=1.2 + top_p=1.0 + top_k=0 gave plenty of
# diversity but blew past the safe vocab — at step 3 we saw 90%+ parse_fail
# with multilingual gibberish and stray special tokens. Pulling back to the
# probe-validated config: still high entropy (probe: ~4 unique plans / K=8),
# but the long-tail of unsafe tokens is truncated so most rollouts stay on
# the JSON rail.
rollout_temperature=1.0                 # was 1.2 — calmer, fewer token-level outliers
rollout_top_p=0.95                      # was 1.0 — drop the bottom 5% (rare unicode etc.)
rollout_top_k=50                        # was 0   — top-50 covers all sensible JSON tokens
rollout_max_new_tokens=512
clip_eps=0.2
kl_beta=0.1                             # was 0.04 - increased to anchor policy near base; prevents skip-collapse

# ---- Reward shaping ----
action_cost=0.1                        # was 0.1 - too high made skip strictly dominate call_wm
parse_fail_penalty=-0.5
# Extra penalty when the policy chose `skip` and the answer is wrong. With
# action_cost=0.02 and ~3 atomic steps a wrong call_wm costs ~0.06; a wrong
# skip without this penalty costs 0, so the model learns to skip even when
# unsure. 0.5 makes a wrong skip strictly worse than the most expensive
# wrong call_wm, restoring the incentive to query WM under uncertainty.
skip_wrong_penalty=0.5

# ---- Optim ----
lr=2e-5                                 # matches the step-140 run (was 5e-5)
per_device_batch_size=1
grad_accum_steps=1                      # matches the step-140 run (was 4); effective batch = 8 GPUs
max_steps=200                           # absolute upper bound; step counter resumes at start_step # best 140
max_grad_norm=1.0

# ---- Resume ----
# Resume LoRA weights from the best-so-far checkpoint. Optimizer state is NOT
# persisted across runs, so AdamW restarts fresh — that's intentional, the LR
# schedule is constant anyway.
resume_adapter=""
start_step=0

# ---- LoRA ----
lora_r=8
lora_alpha=16
lora_dropout=0.05

mkdir -p "$save_dir" "$scratch_dir"

torchrun \
    --nproc_per_node 8 \
    --master_port 29500 \
    training/train_qwen_grpo.py \
    --train_file $train_file \
    --save_dir $save_dir \
    --scratch_dir $scratch_dir \
    \
    --policy_model_name "$policy_model_name" \
    --qa_model_name "gpt-4o" \
    --qa_provider "azure" \
    \
    --num_rollouts $num_rollouts \
    --rollout_temperature $rollout_temperature \
    --rollout_top_p $rollout_top_p \
    --rollout_top_k $rollout_top_k \
    --rollout_max_new_tokens $rollout_max_new_tokens \
    --clip_eps $clip_eps \
    --kl_beta $kl_beta \
    \
    --action_cost $action_cost \
    --parse_fail_penalty $parse_fail_penalty \
    --skip_wrong_penalty $skip_wrong_penalty \
    \
    --lr $lr \
    --per_device_batch_size $per_device_batch_size \
    --grad_accum_steps $grad_accum_steps \
    --max_steps $max_steps \
    --max_grad_norm $max_grad_norm \
    \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    \
    --resume_adapter $resume_adapter \
    --start_step $start_step \
    \
    --prompt_style soft \
    \
    --max_action_ids_cap 6 \
    --max_atomic_actions 6 \
    --sampling_interval_meter 0.25 \
    --sampling_interval_angle 9 \
    \
    --task "img2trajvid_s-prob" \
    --replace_or_include_input True \
    --cfg 4.0 \
    --guider 1 \
    --L_short 576 \
    --num_targets 8 \
    --use_traj_prior True \
    --chunk_strategy interp \
    --frame_interval 3 \
    --max_images 2 \
    \
    --log_every 1 \
    --save_every 10 \
    \
    --wandb_project "$wandb_project" \
    --wandb_entity "$wandb_entity" \
    --wandb_run_name "$wandb_run_name"
