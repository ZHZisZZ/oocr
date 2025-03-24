model_name_or_path=$1
data_config=$2 

WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 \
    accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/two_hop/src/train.py \
    --model_name_or_path ${model_name_or_path} \
    --config_path oocr/two_hop/configs/${data_config}.yaml \
    --output_dir models/${data_config}/$(basename ${model_name_or_path})
