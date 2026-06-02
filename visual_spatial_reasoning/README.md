# Visual Spatial Reasoning — Adaptive Visual Imagination Control (AVIC)

This folder contains the **visual spatial reasoning** experiments for *When and
How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual
Spatial Reasoning*, evaluated on the [SAT](https://huggingface.co/datasets/array/SAT)
dataset. It builds on [MindJourney](https://github.com/UMass-Embodied-AGI/MindJourney).

There are two complementary pieces:

1. **Training-free AVIC** (`pipelines/pipeline_avic.py`) — a closed-source VLM
   (e.g. GPT-4.1 / GPT-4o) adaptively decides *when* and *how much* to imagine
   with the Stable Virtual Camera (SVC) world model via spatial beam search.
2. **RL-trained policy** (same `pipelines/pipeline_avic.py`, run with
   `--policy_model_type qwen2.5vl` + `--policy_lora_ckpt`, plus `training/`) —
   a Qwen2.5-VL policy is trained with **GRPO** to make the imagine / skip +
   action-planning decisions, replacing the prompted closed-source gating with
   a small open policy model.

Both modes share a single pipeline: `pipeline_avic.py` runs the GPT gating by
default (`--policy_model_type gpt`) and switches to the local Qwen-VL policy
when `--policy_model_type qwen3vl|qwen2.5vl` is passed.

---

## Repository structure

```
visual_spatial_reasoning/
├── pipelines/
│   ├── pipeline_baseline.py            # no-world-model baseline (PipelineBase)
│   └── pipeline_avic.py                # unified pipeline: training-free AVIC (GPT gating) AND
│                                       #   RL policy mode (--policy_model_type qwen2.5vl); also the GRPO reward rollout
├── training/                           # GRPO RL training package
│   ├── train_qwen_grpo.py              # torchrun entrypoint (8-GPU online GRPO)
│   ├── qwen_grpo.py                    # LoRA GRPO trainer (PPO-clip + KL-to-ref)
│   ├── wm_qa_rollout.py                # WM + GPT-4o QA rollout -> reward
│   └── probe_policy_diversity.py       # tune sampling so K rollouts stay diverse
├── tools/
│   ├── prescore_train_skip.py          # GPT-4o skip-only prescoring of SAT-train
│   ├── build_balanced_train.py         # build the balanced GRPO train set
│   └── aggregate_chunks.py             # merge parallel eval chunks -> results.json / CSV
├── utils/                              # api.py, args.py, prompt_formatting.py, vlm_wrapper.py, qwen_policy.py, data_process.py, InternVL3.py
├── scripts/                            # ready-to-run drivers (see below)
├── stable_virtual_camera/              # SVC world model (editable install; deps in pyproject.toml)
├── data/                               # SAT splits (val/test) + train_balanced_grpo.json
└── requirements_train.txt              # extra deps for RL training
```

---

## 1. Environment setup

A single Conda environment holds the VLM framework, the SVC world model, and the
RL training stack (the GRPO rollout runs SVC + the Qwen policy + GPT QA together,
so they must share one env):

```bash
conda create -n avic python=3.11 -y
conda activate avic

# CUDA 12.6 builds of PyTorch (adjust to your CUDA)
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
  --extra-index-url https://download.pytorch.org/whl/cu126

# Stable Virtual Camera world model (editable install; deps in pyproject.toml)
pip install -e stable_virtual_camera/

# Extra deps for RL (GRPO) policy training
pip install -r requirements_train.txt
```

Add the repo to `PYTHONPATH` and select the world model before running anything:

```bash
export PYTHONPATH=$PYTHONPATH:./
export WORLD_MODEL_TYPE="svc"
```

> If you hit dependency conflicts and only need *training-free* SVC inference (no
> RL training), you can instead install SVC in its own `python=3.10` env via
> `pip install -e stable_virtual_camera/`.

**Hardware.** SVC needs a large GPU (≈80 GB recommended). GRPO training as shipped
uses **8 GPUs** (one rollout group per GPU). Single-GPU training works by setting
`--nproc_per_node 1` but is slow.

---

## 2. Configure the closed-source VLM (Azure OpenAI)

The training-free pipeline and the GRPO **QA reward** use a GPT-family VLM through
`utils/api.py`. **Do not hardcode keys** — export them in your shell:

```bash
export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
export AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE.cognitiveservices.azure.com/"
```

Supported VLMs: `gpt-4o`, `gpt-4.1`, `o4-mini`, `o1`, or local
`OpenGVLab/InternVL3-8B` / `InternVL3-14B`. You are responsible for any API costs.

The open **policy** model (`Qwen/Qwen2.5-VL-7B-Instruct`) and the SVC weights are
pulled from Hugging Face:

```bash
# SVC weights require approval: https://huggingface.co/stabilityai/stable-virtual-camera
huggingface-cli login
```

---

## 3. Data preparation (SAT)

```bash
python utils/data_process.py --split val      # data/val.json   + data/val/image_*.png
python utils/data_process.py --split test      # data/test.json  + data/test/image_*.png
python utils/data_process.py --split train     # data/train.json + data/train/image_*.png  (needed for RL)
```

`data/val.json` and `data/test.json` are committed; the per-image PNGs are
downloaded by the commands above. For RL you additionally need the `train` split
images.

> **✅ The GRPO training set is already provided — you only need to download the images.**
>
> We ship a ready-to-use, GPT-4o-prescored, balanced GRPO training set at
> [`data/train_balanced_grpo.json`](data/train_balanced_grpo.json) (3,287
> questions; 30% `easy_skip` / 70% `needs_wm`). You do **not** need to run any
> prescoring or balancing yourself. The only required step is to fetch the
> images its `img_paths` point at (`./data/train/image_*.png`):
>
> ```bash
> python utils/data_process.py --split train
> ```
>
> Rebuilding the JSON from scratch is optional — see
> [§5.1](#51-build-the-grpo-training-set-optional).

---

## 4. Training-free AVIC inference

Run from this folder with `PYTHONPATH=./` and `WORLD_MODEL_TYPE=svc` exported.

```bash
# No-world-model baseline
bash scripts/pipeline_baseline_sat_test.sh

# Training-free AVIC (SVC spatial beam search, GPT-4.1 gating)
bash scripts/pipeline_avic.sh
```

Key arguments (full list in `utils/args.py`):

| arg | meaning |
| --- | --- |
| `--vlm_model_name` / `--vlm_qa_model_name` | scoring VLM / answering VLM (`None` ⇒ same as scoring) |
| `--num_questions`, `--split` | number of questions and `val`/`test` |
| `--max_steps_per_question` | beam-search depth (imagination steps) |
| `--num_beams`, `--num_top_candidates` | beam width / candidate count |
| `--helpful_score_threshold`, `--exploration_score_threshold` | gating thresholds |
| `--max_images` | images per question (1–2) |
| `--num_question_chunks`, `--question_chunk_idx` | split questions into parallel chunks |

---

## 5. RL training (GRPO)

The policy is a LoRA-wrapped `Qwen/Qwen2.5-VL-7B-Instruct`. For each training
question we sample **K rollouts**, run the SVC world model + GPT-4o QA to score
each rollout, compute group-relative advantages, and update with a PPO-clip loss
plus a KL penalty to the frozen base policy. Reward shaping penalizes action cost,
parse failures, and confidently-wrong skips.

### 5.1 Build the GRPO training set (optional — JSON already provided)

**You can skip this entire subsection.** We already provide the balanced training
set at `data/train_balanced_grpo.json`; just download the train images (§3) and
go straight to [§5.3](#53-launch-grpo-training). The steps below are only for
rebuilding the set from scratch on different data.

```bash
# (1) GPT-4o "skip-only" prescore of a stratified train sample -> JSONL
python tools/prescore_train_skip.py \
    --num_questions 10000 --concurrency 8 \
    --output data/train_prescored_10k.jsonl

# (2) Bucket into easy_skip / needs_wm and emit a balanced set
python tools/build_balanced_train.py \
    --prescore data/train_prescored_10k.jsonl \
    --sample   data/train_sample_10k.json \
    --output   data/train_balanced_grpo.json \
    --easy_frac 0.3 --needs_wm_frac 0.7
```

### 5.2 (Optional) Probe sampling diversity

GRPO needs reward variance, which needs *plan* variance across the K rollouts.
This grid-searches temperature / top_p / top_k to find a config giving ~3–4
unique plans per K=8 group:

```bash
bash scripts/probe_policy_diversity.sh
```

### 5.3 Launch GRPO training

Set your keys first (the QA reward calls Azure GPT-4o; W&B is optional):

```bash
export AZURE_OPENAI_API_KEY=...      # required (QA reward)
export AZURE_OPENAI_ENDPOINT=...     # required
export WANDB_API_KEY=...             # optional; unset to disable W&B logging
```

Then launch the 8-GPU online GRPO trainer:

```bash
bash scripts/train_qwen_grpo.sh
```

This is a thin wrapper over:

```bash
torchrun --nproc_per_node 8 training/train_qwen_grpo.py --train_file data/train_balanced_grpo.json ...
```

Key knobs in `scripts/train_qwen_grpo.sh` (and `training/train_qwen_grpo.py --help`):

| group | args |
| --- | --- |
| GRPO | `--num_rollouts` (K), `--rollout_temperature/top_p/top_k`, `--clip_eps`, `--kl_beta` |
| reward shaping | `--action_cost`, `--parse_fail_penalty`, `--skip_wrong_penalty` |
| optim | `--lr`, `--per_device_batch_size`, `--grad_accum_steps`, `--max_steps`, `--max_grad_norm` |
| LoRA | `--lora_r`, `--lora_alpha`, `--lora_dropout` |
| resume | `--resume_adapter <dir>`, `--start_step <N>` |
| prompt | `--prompt_style {strict,soft}` |
| WM/action | `--max_action_ids_cap`, `--max_atomic_actions`, `--sampling_interval_meter/angle`, SVC flags (`--task`, `--cfg`, `--guider`, `--L_short`, `--num_targets`, `--use_traj_prior`, `--chunk_strategy`) |

LoRA adapters are saved every `--save_every` steps to
`<save_dir>/adapter_step<N>/` (rank 0 only). The default effective batch is
8 GPUs × 1 × 1 = 8 questions/step.

### 5.4 Evaluate trained checkpoints

Run inference with a trained policy by pointing `--policy_lora_ckpt` at an
`adapter_step*` directory:

```bash
# Single config, 8-GPU parallel over question chunks
#   (edit policy_lora_ckpt / output_dir at the top of the script first)
bash scripts/inference_avic_rl_parallel.sh

# Or sweep every adapter_step* in a run dir and emit a per-step accuracy CSV
bash scripts/batch_eval_ckpts.sh nips_results/<run_dir>            # all steps
bash scripts/batch_eval_ckpts.sh nips_results/<run_dir> 10 30 100  # specific steps
```

`batch_eval_ckpts.sh` writes `nips_results/eval_<run>_summary.csv` (one row per
checkpoint: step, total, overall + per-question-type accuracy). Tunables:
`BATCH_NUM_CHUNKS`, `BATCH_GPUS`, `BATCH_NUM_QUESTIONS`, `BATCH_DATASET`, `BATCH_TAG`.

### 5.5 Train then evaluate in one shot

```bash
bash scripts/train_then_eval.sh                 # train, then eval all checkpoints
bash scripts/train_then_eval.sh 10 30 60 100    # eval only these steps
EVAL_NUM_QUESTIONS=50 bash scripts/train_then_eval.sh
```

---

## Released checkpoint (best setting)

Our best-performing policy is the **`adapter_step140`** LoRA adapter
(`Qwen/Qwen2.5-VL-7B-Instruct` base). It beats the base policy by **+6 points
overall** on the SAT test set.

**Download** (coming soon — update the repo id once uploaded):

```bash
# https://huggingface.co/<HF_USERNAME>/AVIC-Qwen2.5-VL-7B-policy
huggingface-cli download <HF_USERNAME>/AVIC-Qwen2.5-VL-7B-policy \
    --local-dir checkpoints/AVIC-Qwen2.5-VL-7B-policy
# the LoRA adapter lives at checkpoints/AVIC-Qwen2.5-VL-7B-policy/adapter_step140
```

### Training setting that produced `adapter_step140`

Driver: `scripts/train_qwen_grpo.sh` (8-GPU `torchrun`, effective batch = 8
questions/step). To reproduce from scratch, set `resume_adapter=""` and
`start_step=0` in the script and train to step 140.

| group | setting |
| --- | --- |
| base policy | `Qwen/Qwen2.5-VL-7B-Instruct` (LoRA) |
| train data | `data/train_balanced_grpo.json` (30% easy_skip / 70% needs_wm) |
| GRPO | `num_rollouts=16`, `rollout_temperature=1.0`, `rollout_top_p=0.95`, `rollout_top_k=50`, `rollout_max_new_tokens=512`, `clip_eps=0.2`, `kl_beta=0.1` |
| reward shaping | `action_cost=0.1`, `parse_fail_penalty=-0.5`, `skip_wrong_penalty=0.5` |
| optim | `lr=2e-5`, `per_device_batch_size=1`, `grad_accum_steps=1`, `max_grad_norm=1.0`, 8 GPUs |
| LoRA | `lora_r=8`, `lora_alpha=16`, `lora_dropout=0.05` |
| prompt | `prompt_style=soft` |
| WM / action | `max_action_ids_cap=6`, `max_atomic_actions=6`, `sampling_interval_meter=0.25`, `sampling_interval_angle=9` |
| SVC | `task=img2trajvid_s-prob`, `cfg=4.0`, `guider=1`, `L_short=576`, `num_targets=8`, `use_traj_prior=True`, `chunk_strategy=interp`, `frame_interval=3`, `max_images=2` |

### Eval setting for `adapter_step140`

Driver: `scripts/inference_avic_rl_parallel.sh` (set
`policy_lora_ckpt=<...>/adapter_step140`) or
`scripts/batch_eval_ckpts.sh <run_dir> 140`. Full SAT test set, GPT-4o as the QA
model.

| group | setting |
| --- | --- |
| split / size | `--split test`, `--num_questions 150` |
| QA VLM | `--provider azure`, `--vlm_model_name gpt-4o`, `--vlm_qa_model_name None` |
| policy | `--policy_model_type qwen2.5vl`, `--policy_model_name Qwen/Qwen2.5-VL-7B-Instruct`, `--policy_lora_ckpt .../adapter_step140` |
| policy sampling | `--policy_temperature 0.7`, `--policy_top_p 1.0`, `--policy_max_new_tokens 512`, `--num_policy_samples 5` |
| search | `--scaling_strategy spatial_beam_search`, `--max_steps_per_question 3`, `--num_beams 3`, `--num_top_candidates 6`, `--max_wm_candidates 5`, `--max_action_ids_cap 6`, `--max_tries_gpt 4` |
| gating | `--helpful_score_threshold 8`, `--exploration_score_threshold 8` |
| views / WM | `--max_images 2`, `--num_frames 9`, `--frame_interval 3`, `--sampling_interval_angle 9`, `--sampling_interval_meter 0.25`, `--fixed_rotation_magnitudes 27`, `--fixed_forward_magnitudes 0.75` |
| SVC | `--task img2trajvid_s-prob`, `--cfg 4.0`, `--guider 1`, `--L_short 576`, `--num_targets 8`, `--use_traj_prior True`, `--chunk_strategy interp` |

---

## 6. Results and logs

Outputs go under `--output_dir`:

- `results.json` — overall accuracy, per-type accuracy, skipped indices, parsing stats
- `<qid>/` — starting image(s) and `gpt.json` / `timing.json` per-question logs

For parallel runs each `--question_chunk_idx` writes to its own
`question_chunk_*/` subdir; merge them with `tools/aggregate_chunks.py`
(invoked automatically by `batch_eval_ckpts.sh`).

---

## Notes

- This is research code under active development; interfaces and scripts may change.
- Scripts read API keys from the environment and ship with placeholders only —
  never commit real keys.
- SVC weights require Hugging Face approval and a large-VRAM GPU.
- For arguments, see `utils/args.py`; for pipeline behavior, see the docstrings
  in `pipelines/` and `training/`.
