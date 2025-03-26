model_name_or_path=$1
data_config=$2

for learning_rate in 1e-6 3e-6 1e-5 3e-5; do
    output_base_dir=models/${data_config}/$(basename ${model_name_or_path})/lr-${learning_rate}
    for seed in $(seq 0 5); do
        WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
            accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/two_hop/src/train.py \
            --model_name_or_path ${model_name_or_path} \
            --data_config_path oocr/two_hop/configs/${data_config}.yaml \
            --learning_rate ${learning_rate} \
            --seed ${seed} \
            --output_dir ${output_base_dir}/seed-${seed} &
        sleep 1
    done
    wait
    python oocr/two_hop/src/plot.py --output_base_dir ${output_base_dir}
done
