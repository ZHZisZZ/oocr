[Extractive Structures Learned in Pretraining Enable Generalization on Finetuned Facts](https://arxiv.org/pdf/2412.04614)

```bash
# see `models/city_first_hop/Meta-Llama-3-8B/*/plot.png`
sh oocr/two_hop/scripts/train.sh /mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B city_first_hop

sh oocr/two_hop/scripts/train.sh /mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-14B city_first_hop deepspeed_zero2 8

sh oocr/two_hop/scripts/train.sh /mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-32B city_first_hop deepspeed_zero3 8
```
