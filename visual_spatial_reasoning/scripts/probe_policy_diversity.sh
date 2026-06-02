#!/usr/bin/env bash
# Standalone Qwen3-VL policy sampling-diversity tuner.
# No WM, no QA, no training. Just: sample K plans for each (T, top_p, top_k)
# combo on a few real training questions and report unique-plan counts.
#
# Why: GRPO needs reward variance, which needs *plan* variance across the K
# rollouts. If 8/8 rollouts produce the same plan, advantages collapse to 0
# and the policy can't learn. This script grids over sampling configs so you
# can pick one that gives ~3-4 unique plans per K=8 group.

export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Probe ~4 candidate models (and optionally a "soft" CoT-style prompt) across
# a small T/top_p/top_k grid. Pick the (model, prompt, config) triple with the
# highest avg_unique_plans before launching real GRPO.
#
# WARNING: each model is loaded one at a time and frees the GPU before the
# next; total wall time scales with len(models) * len(grid_cells).

# models="Qwen/Qwen3-VL-4B-Instruct,Qwen/Qwen3-VL-8B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct"
models="Qwen/Qwen2.5-VL-7B-Instruct"
prompt_style="strict"     # "strict" (training default) or "soft" (CoT-style)
input_file="data/train.json"
num_questions=5
K=16
max_new_tokens=512

# Smaller grid for multi-model: 2 T * 2 top_p * 1 top_k = 4 cells per model.
temperatures="1.0,1.5"
top_p_values="1.0,0.95"
top_k_values="50"
repetition_penalty=1.0    # try 1.1-1.3 if outputs look loop-y

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python training/probe_policy_diversity.py \
    --models "$models" \
    --prompt_style "$prompt_style" \
    --input_file "$input_file" \
    --num_questions $num_questions \
    --K $K \
    --max_new_tokens $max_new_tokens \
    --temperatures "$temperatures" \
    --top_p_values "$top_p_values" \
    --top_k_values "$top_k_values" \
    --repetition_penalty $repetition_penalty
