export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./
num_questions=150
scaling_strategy="spatial_beam_search"
question_type="None"
vlm_model_name="gpt-4.1"
vlm_qa_model_name=None # pass None will be interpreted as "None" anyway; None means qa_vlm_model_name is same as vlm_model_name
max_images=2
max_steps=3 # Only 1 step for test runs
dataset_type="test" # choose from "val", "test"
input_dir="data"
output_dir="results/exp1/"

cmd="CUDA_VISIBLE_DEVICES=3 python pipelines/pipeline_avic.py \
  \
  --vlm_model_name=$vlm_model_name \
  --vlm_qa_model_name=$vlm_qa_model_name \
  --num_questions $num_questions \
  --output_dir $output_dir \
  --input_dir $input_dir \
  --scaling_strategy $scaling_strategy \
  --question_type $question_type \
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
  --num_question_chunks 1 \
  --question_chunk_idx 0 \
  \
  --task "img2trajvid_s-prob" \
  --replace_or_include_input True\
  --cfg 4.0
  --guider 1  \
  --L_short 576  \
  --num_targets 8  \
  --use_traj_prior True \
  --chunk_strategy "interp"
  "
echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"