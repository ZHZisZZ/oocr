model_name_or_path=$1
data_config=$2
accelerate_configs=${3:-"single_gpu"}  # Default to single gpu
gpu=${4:-1}  # Default to 1 GPU

for learning_rate in 1e-6 3e-6 1e-5 3e-5; do
    output_base_dir=models/${data_config}/$(basename ${model_name_or_path})/lr-${learning_rate}
    for seed in $(seq 0 5); do
        WANDB_MODE=offline WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:${gpu} --cpus-per-task=16 --time=3000 \
            accelerate launch --config_file configs/accelerate_configs/${accelerate_configs}.yaml --num_processes ${gpu} oocr/two_hop/src/train.py \
            --model_name_or_path ${model_name_or_path} \
            --data_config_path oocr/two_hop/configs/${data_config}.yaml \
            --per_device_train_batch_size $((8 / gpu)) \
            --learning_rate ${learning_rate} \
            --seed ${seed} \
            --output_dir ${output_base_dir}/seed-${seed} &
        sleep 1
    done
    wait
    python oocr/two_hop/src/plot.py --output_base_dir ${output_base_dir}
done
