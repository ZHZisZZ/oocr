# for debug, and then sh test.sh
# WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
#     accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/two_hop/src/train.py \
#     --model_name_or_path ${model_name_or_path} \
#     --data_config_path oocr/two_hop/configs/${data_config}.yaml \
#     --eval_strategy "no" \
#     --save_strategy "epoch" \ 
#     --seed 0 \
#     --output_dir models/${data_config}/$(basename ${model_name_or_path})/seed-0

# PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
#     python oocr/two_hop/src/test.py \
#     --model_name_or_path models/${data_config}/$(basename ${model_name_or_path})/seed-0/checkpoint-0 \
#     --data_config_path oocr/two_hop/configs/${data_config}.yaml \
#     --template "all"

# python plot.py --output_base_dir models/${data_config}/$(basename ${model_name_or_path})
