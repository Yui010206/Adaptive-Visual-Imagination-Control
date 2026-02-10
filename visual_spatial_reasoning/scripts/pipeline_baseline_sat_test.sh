export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./
num_questions=10
scaling_strategy="spatial_beam_search"
question_type="None"
vlm_model_name="gpt-4.1"
vlm_qa_model_name=None # pass None will be interpreted as "None" anyway; None means qa_vlm_model_name is same as vlm_model_name
max_images=2
max_steps_per_question=15
dataset_type="test" # choose from "val", "test"
input_dir="data"
output_dir="results/exp/"

cmd="python pipelines/pipeline_baseline.py \
  \
  --vlm_model_name=$vlm_model_name \
  --vlm_qa_model_name=$vlm_qa_model_name \
  --num_questions=$num_questions \
  --max_steps_per_question=$max_steps_per_question \
  --output_dir $output_dir \
  --input_dir $input_dir \
  --question_type None \
  --max_images $max_images \
  --max_tries_gpt 5 \
  --split test \
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