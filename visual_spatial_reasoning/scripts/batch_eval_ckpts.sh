#!/usr/bin/env bash
# Batch-evaluate a directory of LoRA checkpoints from GRPO training.
# For each adapter_step* subdir, runs the same parallel inference as
# scripts/inference_avic_rl_parallel.sh and aggregates results.
#
# Usage:
#   bash scripts/batch_eval_ckpts.sh <ckpt_dir> [step ...]
#
# Examples:
#   # All adapter_step* in the dir, in numeric order
#   bash scripts/batch_eval_ckpts.sh nips_results/grpo_qwen25vl_7b_8gpu_soft+skip_pen
#
#   # Only specific steps
#   bash scripts/batch_eval_ckpts.sh nips_results/grpo_qwen25vl_7b_8gpu_soft+skip_pen 10 30 60 100
#
# Tunable env vars (all optional):
#   BATCH_NUM_CHUNKS=4               # parallel chunks per ckpt (= GPUs used)
#   BATCH_GPUS="0 1 2 3"             # GPU ids; must list NUM_CHUNKS values
#   BATCH_NUM_QUESTIONS=150          # how many test questions to score
#   BATCH_DATASET=test               # val | test
#   BATCH_TAG=""                     # extra suffix tacked onto output_dir
#
# Output:
#   - per-ckpt: nips_results/eval_<ckpt_basename>_step<N>/_spatial_beam_search_qc<NUM_CHUNKS>/
#       contains question_chunk_*/ + a merged results.json
#   - global:   nips_results/eval_<ckpt_basename>_summary.csv
#       one row per ckpt: step,total,acc_all,<per-qtype>...
#   - logs:     <output_dir>/_qc<NUM_CHUNKS>/logs/chunk_<i>_gpu<g>.log

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <ckpt_dir> [step_numbers...]" >&2
  exit 1
fi

CKPT_DIR="${1%/}"
shift
EXPLICIT_STEPS=("$@")

if [ ! -d "$CKPT_DIR" ]; then
  echo "[fatal] ckpt dir does not exist: $CKPT_DIR" >&2
  exit 1
fi

# Discover steps if none given.
if [ ${#EXPLICIT_STEPS[@]} -eq 0 ]; then
  mapfile -t STEPS < <(
    for d in "$CKPT_DIR"/adapter_step*; do
      [ -d "$d" ] || continue
      basename "$d" | sed 's/adapter_step//'
    done | sort -n
  )
  if [ ${#STEPS[@]} -eq 0 ]; then
    echo "[fatal] no adapter_step* subdirs in $CKPT_DIR" >&2
    exit 1
  fi
else
  STEPS=("${EXPLICIT_STEPS[@]}")
fi

ckpt_basename=$(basename "$CKPT_DIR")
echo "[plan] ckpt_dir = $CKPT_DIR"
echo "[plan] eval ${#STEPS[@]} steps: ${STEPS[*]}"

# ---- Common config ----
export WORLD_MODEL_TYPE=svc
export PYTHONPATH=$PYTHONPATH:./
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-YOUR_AZURE_OPENAI_API_KEY}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://YOUR_RESOURCE.cognitiveservices.azure.com/}"

provider=azure
vlm_model_name=gpt-4o
vlm_qa_model_name=None
policy_model_type=qwen2.5vl
policy_model_name="Qwen/Qwen2.5-VL-7B-Instruct"
policy_temperature=0.7
policy_top_p=1.0
policy_max_new_tokens=512

num_questions=${BATCH_NUM_QUESTIONS:-150}
scaling_strategy=spatial_beam_search
question_type=None
helpful_score_threshold=8
exploration_score_threshold=8
max_images=2
max_steps=3
num_policy_samples=5
max_wm_candidates=5
max_action_ids_cap=6

dataset_type=${BATCH_DATASET:-test}
input_dir=data

# ---- Parallelism ----
num_question_chunks=${BATCH_NUM_CHUNKS:-4}
GPUS_STR="${BATCH_GPUS:-0 1 2 3}"
read -ra gpus <<< "$GPUS_STR"
if [ ${#gpus[@]} -ne "$num_question_chunks" ]; then
  echo "[fatal] BATCH_GPUS has ${#gpus[@]} ids but BATCH_NUM_CHUNKS=$num_question_chunks" >&2
  exit 1
fi

extra_tag="${BATCH_TAG:+_${BATCH_TAG}}"
summary_csv="nips_results/eval_${ckpt_basename}${extra_tag}_summary.csv"
mkdir -p "$(dirname "$summary_csv")"
echo "step,total,acc_all,ego_movement,obj_movement,goal_aim,action_conseq,perspective" > "$summary_csv"

n_done=0
n_skipped=0
for step in "${STEPS[@]}"; do
  ckpt="$CKPT_DIR/adapter_step${step}"
  if [ ! -d "$ckpt" ]; then
    echo "[skip] $ckpt missing"; n_skipped=$((n_skipped+1)); continue
  fi

  out_dir="nips_results/eval_${ckpt_basename}${extra_tag}_step${step}/"
  qc_dir="${out_dir%/}_qc${num_question_chunks}"
  log_dir="${qc_dir}/logs"
  mkdir -p "$log_dir"

  echo
  echo "============================================================"
  echo " step=$step  ckpt=$ckpt"
  echo " out=$out_dir"
  echo "============================================================"

  pids=()
  for idx in "${!gpus[@]}"; do
    gpu_id=${gpus[$idx]}
    log_file="${log_dir}/chunk_${idx}_gpu${gpu_id}.log"
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python pipelines/pipeline_avic.py \
      --provider=$provider \
      --vlm_model_name=$vlm_model_name \
      --vlm_qa_model_name=$vlm_qa_model_name \
      --num_questions $num_questions \
      --output_dir $out_dir \
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
      --num_policy_samples $num_policy_samples \
      --max_wm_candidates $max_wm_candidates \
      --max_action_ids_cap $max_action_ids_cap \
      --policy_model_type $policy_model_type \
      --policy_model_name $policy_model_name \
      --policy_temperature $policy_temperature \
      --policy_top_p $policy_top_p \
      --policy_max_new_tokens $policy_max_new_tokens \
      --policy_lora_ckpt $ckpt \
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
    pids+=("$!")
  done

  echo "Launched ${#pids[@]} chunks (pids: ${pids[*]}). Waiting..."
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[!] pid $pid exited non-zero"
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "[!] step=$step had failures; logs in $log_dir"
    n_skipped=$((n_skipped+1))
    continue
  fi

  # Aggregate + emit one CSV row.
  python tools/aggregate_chunks.py "$qc_dir" --csv --label "$step" >> "$summary_csv"
  n_done=$((n_done+1))
done

echo
echo "============================================================"
echo " Done: $n_done evaluated, $n_skipped skipped/failed"
echo " Summary CSV: $summary_csv"
echo "============================================================"
column -t -s, "$summary_csv"
