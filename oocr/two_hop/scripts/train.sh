#!/bin/bash

data_config=city_first_hop
model_name_list=('/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B' '/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-7B')
learning_rates=(1e-6 3e-6 1e-5 3e-5)
epochs=(4 8 12 16)
sleep_time=1

for model_name_or_path in "${model_name_list[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        for num_epochs in "${epochs[@]}"; do
            output_dir="/mnt/lustrenew/mllm_safety-shared/tmp/lingjie/checkpoints/$(basename ${model_name_or_path})/oocr/${data_config}_lr${learning_rate}_ep${num_epochs}/"
            echo $output_dir

            WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
                accelerate launch --config_file /mnt/petrelfs/chenlingjie/oocr/oocr/two_hop/configs/accelerate_configs/single_gpu.yaml \
                two_hop/src/train.py \
                --model_name_or_path ${model_name_or_path} \
                --config_path two_hop/configs/${data_config}.yaml \
                --num_train_epochs ${num_epochs} \
                --learning_rate ${learning_rate} \
                --output_dir ${output_dir}&
            # break
            sleep ${sleep_time}
        done
    done
done


# learning_rate=3e-6
# echo /mnt/lustrenew/mllm_safety-shared/tmp/lingjie/checkpoints/$(basename ${model_name_or_path})/oocr/${data_config}_${learning_rate}/

# WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
#     accelerate launch --config_file /mnt/petrelfs/chenlingjie/oocr/oocr/two_hop/configs/accelerate_configs/single_gpu.yaml \
#     two_hop/src/train.py \
#     --model_name_or_path ${model_name_or_path} \
#     --config_path two_hop/configs/${data_config}.yaml \
#     --num_train_epochs num_epochs \
#     --learning_rate ${learning_rate} \
#     --output_dir /mnt/lustrenew/mllm_safety-shared/tmp/lingjie/checkpoints/$(basename ${model_name_or_path})/oocr/${data_config}_${learning_rate}/