#!/usr/bin/env bash
# Launch pipeline_avic.py (RL-policy mode) over 8 GPUs in parallel.
# Each GPU handles 1/8 of the questions; outputs go into a per-chunk subdir
# (the pipeline auto-suffixes output_dir with /question_chunk_<idx>).

export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./

# ---- API credentials (set in env; do NOT hardcode real keys) ----
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-YOUR_AZURE_OPENAI_API_KEY}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://YOUR_RESOURCE.cognitiveservices.azure.com/}"
# export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

# ---- Models ----
provider="azure"
vlm_model_name="gpt-4o"
vlm_qa_model_name=None

policy_model_type="qwen2.5vl"
policy_model_name="Qwen/Qwen2.5-VL-7B-Instruct"
# Optional LoRA adapter from GRPO training. Leave empty to use the base model.
# Example: nips_results/grpo_qwen25vl_7b_8gpu/adapter_step100  (or .../adapter_final)
# Optional LoRA adapter from GRPO training. Empty = base model.
# Best released checkpoint: checkpoints/AVIC-Qwen2.5-VL-7B-policy/adapter_step140
policy_lora_ckpt=""
policy_temperature=0.7
policy_top_p=1.0
policy_max_new_tokens=512

# ---- Pipeline config ----
num_questions=150
scaling_strategy="spatial_beam_search"
question_type="None"
helpful_score_threshold=8
exploration_score_threshold=8
max_images=2
max_steps=3
num_policy_samples=5
max_wm_candidates=5
max_action_ids_cap=6

dataset_type="test"
input_dir="data"
output_dir="nips_results/qwen_2.5_policy_gpt4o_qa_soft+skip_pen+0.1_ac_new_130/"

# ---- Parallel launch ----
num_question_chunks=8
gpus=(0 1 2 3 4 5 6 7)        # GPU id for each chunk index
log_dir="${output_dir%/}_qc${num_question_chunks}/logs"
mkdir -p "$log_dir"

# Compose optional LoRA flag once.
if [ -n "$policy_lora_ckpt" ]; then
  lora_arg="--policy_lora_ckpt $policy_lora_ckpt"
  echo "[Policy] using LoRA adapter: $policy_lora_ckpt"
else
  lora_arg=""
  echo "[Policy] no LoRA adapter (using base $policy_model_name)"
fi

pids=()
for idx in "${!gpus[@]}"; do
  gpu_id=${gpus[$idx]}
  log_file="${log_dir}/chunk_${idx}_gpu${gpu_id}.log"

  cmd="CUDA_VISIBLE_DEVICES=$gpu_id python pipelines/pipeline_avic.py \
    --provider=$provider \
    --vlm_model_name=$vlm_model_name \
    --vlm_qa_model_name=$vlm_qa_model_name \
    --num_questions $num_questions \
    --output_dir $output_dir \
    --input_dir $input_dir \
    --scaling_strategy $scaling_strategy \
    --question_type $question_type \
    --helpful_score_threshold $helpful_score_threshold \
    --exploration_score_threshold $exploration_score_threshold \
    --max_images $max_images \
    --sampling_interval_angle 9 \
    --sampling_interval_meter 0.25 \
    --fixed_rotation_magnitudes 27 \
    --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question $max_steps \
    --num_top_candidates 6 \
    --num_beams 3 \
    --max_tries_gpt 4 \
    --num_frames 9 \
    --frame_interval 3 \
    --max_inference_batch_size 1 \
    --split $dataset_type \
    --num_question_chunks $num_question_chunks \
    --question_chunk_idx $idx \
    \
    --num_policy_samples $num_policy_samples \
    --max_wm_candidates $max_wm_candidates \
    --max_action_ids_cap $max_action_ids_cap \
    \
    --policy_model_type $policy_model_type \
    --policy_model_name $policy_model_name \
    --policy_temperature $policy_temperature \
    --policy_top_p $policy_top_p \
    --policy_max_new_tokens $policy_max_new_tokens \
    $lora_arg \
    \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --cfg 4.0 \
    --guider 1 \
    --L_short 576 \
    --num_targets 8 \
    --use_traj_prior True \
    --chunk_strategy interp"

  echo "[chunk $idx -> GPU $gpu_id] log: $log_file"
  eval "$cmd" > "$log_file" 2>&1 &
  pids+=($!)
done

echo "Launched ${#pids[@]} chunks (pids: ${pids[*]}). Waiting..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[!] pid $pid exited non-zero"
    fail=1
  fi
done

echo -ne "-------------------- All chunks finished (fail=$fail) --------------------\n\n"
exit $fail
