export PYTHONPATH=$(pwd):$PYTHONPATH
DATA_ROOT=[Your Data Root Directory]
outdir=${DATA_ROOT}/exp/

flag="--root_dir ${DATA_ROOT}
      --img_root [Your Image Root Directory]
      --split MapGPT_72_scenes_processed
      --start 0
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4o
      --response_format json
      --max_tokens 1000
      "

CUDA_VISIBLE_DEVICES=0 python vln/main_gpt.py $flag