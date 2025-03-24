model_name_or_path=$1
data_config=$2

PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 \
    python oocr/two_hop/src/test.py \
    --model_name_or_path ${model_name_or_path} \
    --config_path oocr/two_hop/configs/${data_config}.yaml
