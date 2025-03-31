# gpu=2

# WANDB_MODE=disabled WANDB_PROJECT=mcq PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:2 --cpus-per-task=16 --time=300 \
#     accelerate launch --config_file configs/accelerate_configs/deepspeed_zero2.yaml --num_processes 2 oocr/mcq/src/train.py \
#     --num_proc 1 --save_strategy "steps"