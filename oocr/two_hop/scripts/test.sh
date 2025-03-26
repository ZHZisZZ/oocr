
learning_rates=(1e-6 3e-6 1e-5 3e-5)
epochs=(4 8 12 16)
data_config="city_first_hop"
# model_name_or_path='/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-7B'
model_name_or_path='/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B'
sleep_time=0.5
results_dir=two_hop/$(basename ${model_name_or_path})-sft-evaluation/rank_json
images_dir=two_hop/$(basename ${model_name_or_path})-sft-evaluation/plot
mkdir -p "$results_dir"
mkdir -p "$images_dir"

for learning_rate in "${learning_rates[@]}"; do
    for num_train_epoches in "${epochs[@]}"; do
    model_finetuned_dir="/mnt/lustrenew/mllm_safety-shared/tmp/lingjie/checkpoints/$(basename ${model_name_or_path})/oocr/${data_config}_lr${learning_rate}_ep${num_train_epoches}/"
    
    results_file="${results_dir}/${data_config}_lr${learning_rate}_ep${num_train_epoches}.json"
    image_save_path="${images_dir}/${data_config}_lr${learning_rate}_ep${num_train_epoches}.png"

    # Ensure results file exists
    if [ ! -f "$results_file" ]; then
        echo "{}" > "$results_file"
    fi

    for dir in ${model_finetuned_dir}/*/; do 
        if [ -d "$dir" ]; then
            checkpoint_name=$(basename "$dir")
            echo $checkpoint_name
            # Check if this checkpoint is already in results.json
            if grep -q "\"$checkpoint_name\"" "$results_file"; then
                echo "Skipping already tested checkpoint: $checkpoint_name"
                continue
            fi

            echo "Processing directory: $dir"
            PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
            python two_hop/src/test.py \
            --model_name_or_path ${dir} \
            --config_path two_hop/configs/${data_config}.yaml \
            --save_path ${results_file}&
        fi
        sleep ${sleep_time}
    done

    wait
    if [ ! -f "$image_save_path" ]; then
        echo "Image file does not exist. Running the script..."
        PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --cpus-per-task=16 --time=3000 \
            python two_hop/src/get_stats.py \
                --num_train_epochs ${num_train_epoches} \
                --stat_file_path ${results_file} \
                --image_save_path ${image_save_path}
    else
    echo "Skipping execution: $image_save_path already exists."
    fi

    done
done




# model_finetuned_dir="/mnt/lustrenew/mllm_safety-shared/tmp/lingjie/checkpoints/Meta-Llama-3-8B/oocr/city_first_hop_lr1e-5_ep4/"
# data_config="city_first_hop"
# num_train_epoches=4
# learning_rate=1e-5
# results_file="two_hop/results/rank_json/${data_config}_lr${learning_rate}_ep${num_train_epoches}.json"
# echo ${results_file}
# image_save_path="two_hop/results/plot/${data_config}_lr${learning_rate}_ep${num_train_epoches}.png"

# # Ensure results file exists
# if [ ! -f "$results_file" ]; then
#     echo "{}" > "$results_file"
# fi

# for dir in ${model_finetuned_dir}/*/; do 
#     if [ -d "$dir" ]; then
#         checkpoint_name=$(basename "$dir")
#         echo $checkpoint_name
#         # Check if this checkpoint is already in results.json
#         if grep -q "\"$checkpoint_name\"" "$results_file"; then
#             echo "Skipping already tested checkpoint: $checkpoint_name"
#             continue
#         fi

#         echo "Processing directory: $dir"
#         PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
#         python two_hop/src/test.py \
#         --model_name_or_path ${dir} \
#         --config_path two_hop/configs/${data_config}.yaml \
#         --save_path ${results_file}&
#     fi
#     sleep 0.5
# done


# PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --cpus-per-task=16 --time=3000 \
#     python two_hop/src/get_stats.py \
#         --num_train_epochs ${num_train_epoches} \
#         --stat_file_path ${results_file} \
#         --image_save_path ${image_save_path}