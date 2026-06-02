#!/usr/bin/env bash
# End-to-end driver: launch GRPO training (8 GPUs), then as soon as training
# returns, evaluate every saved adapter_step* with batch_eval_ckpts.sh on
# the same 8 GPUs.
#
# Both phases use all 8 GPUs sequentially -- they don't overlap, so there is
# no contention. If training crashes after some checkpoints landed, eval
# still runs on whatever was saved (training failures don't abort eval).
#
# Usage:
#   bash scripts/train_then_eval.sh
#
# Override which steps to evaluate (otherwise: all adapter_step*):
#   bash scripts/train_then_eval.sh 10 30 60 100
#
# Override the eval question count (default 150 = full test set):
#   EVAL_NUM_QUESTIONS=50 bash scripts/train_then_eval.sh
#
# Pin a tag onto eval output dirs (avoids clobbering prior runs):
#   EVAL_TAG=v3 bash scripts/train_then_eval.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_SCRIPT="scripts/train_qwen_grpo.sh"
EVAL_SCRIPT="scripts/batch_eval_ckpts.sh"

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[fatal] $TRAIN_SCRIPT not found" >&2; exit 1
fi
if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "[fatal] $EVAL_SCRIPT not found" >&2; exit 1
fi

# ---- 1. Discover save_dir from the train script ----
# The train script defines `save_dir=...`; we read it instead of hard-coding
# so user tweaks (e.g. renaming the run) propagate automatically.
SAVE_DIR=$(grep -E '^[[:space:]]*save_dir=' "$TRAIN_SCRIPT" \
           | head -1 \
           | sed -E 's/^[[:space:]]*save_dir=//' \
           | sed -E 's/[[:space:]]+#.*$//' \
           | tr -d '"' | tr -d "'")
if [ -z "$SAVE_DIR" ]; then
  echo "[fatal] could not parse save_dir from $TRAIN_SCRIPT" >&2; exit 1
fi
echo "[plan] training save_dir = $SAVE_DIR"

EXPLICIT_STEPS=("$@")
if [ ${#EXPLICIT_STEPS[@]} -gt 0 ]; then
  echo "[plan] eval steps (override) = ${EXPLICIT_STEPS[*]}"
else
  echo "[plan] eval steps = ALL adapter_step* under $SAVE_DIR"
fi

# ---- 2. Run training (8 GPUs, foreground) ----
echo
echo "============================================================"
echo " PHASE 1/2  Training: $TRAIN_SCRIPT"
echo "============================================================"
TRAIN_LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$TRAIN_LOG")"
echo "[train] log: $TRAIN_LOG"

# Don't bail the whole driver if training exits non-zero -- partial
# checkpoints are still worth evaluating (e.g. the rank-mismatch barrier
# crash that happens at end-of-training but has all checkpoints saved).
set +e
bash "$TRAIN_SCRIPT" 2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e
echo "[train] exit code: $TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "[train] non-zero exit; checking whether any checkpoint landed..."
fi

# Sanity: at least one adapter_step* must exist before eval has anything to do
shopt -s nullglob
ckpts=("$SAVE_DIR"/adapter_step*)
shopt -u nullglob
if [ ${#ckpts[@]} -eq 0 ]; then
  echo "[fatal] no adapter_step* in $SAVE_DIR; nothing to evaluate" >&2
  exit 1
fi
echo "[train] found ${#ckpts[@]} checkpoints under $SAVE_DIR"

# ---- 3. Run batch eval (8 GPUs, sequential per ckpt) ----
echo
echo "============================================================"
echo " PHASE 2/2  Batch eval: $EVAL_SCRIPT"
echo "============================================================"
EVAL_LOG="logs/eval_$(date +%Y%m%d_%H%M%S).log"
echo "[eval] log: $EVAL_LOG"

export BATCH_NUM_CHUNKS=8
export BATCH_GPUS="0 1 2 3 4 5 6 7"
[ -n "$EVAL_NUM_QUESTIONS" ] && export BATCH_NUM_QUESTIONS="$EVAL_NUM_QUESTIONS"
[ -n "$EVAL_TAG" ]            && export BATCH_TAG="$EVAL_TAG"

bash "$EVAL_SCRIPT" "$SAVE_DIR" "${EXPLICIT_STEPS[@]}" 2>&1 | tee "$EVAL_LOG"

echo
echo "============================================================"
echo " Done. Train log: $TRAIN_LOG"
echo " Eval  log: $EVAL_LOG"
echo " Per-ckpt summary CSV is printed at the end of the eval log."
echo "============================================================"
