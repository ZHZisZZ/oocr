# sh oocr/two_hop/scripts/dev.sh /mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B city_first_hop

model_name_or_path=$1
data_config=$2

WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
    accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/two_hop/src/train.py \
    --model_name_or_path ${model_name_or_path} \
    --data_config_path oocr/two_hop/configs/${data_config}.yaml \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --seed 0 \
    --output_dir models/tmp

# evaluate a specific checkpoint
PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
    python oocr/two_hop/src/test.py \
    --model_name_or_path models/tmp/checkpoint-3 \
    --data_config_path oocr/two_hop/configs/${data_config}.yaml \
    --template "all"
