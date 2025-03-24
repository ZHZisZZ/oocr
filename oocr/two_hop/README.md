[Extractive Structures Learned in Pretraining Enable Generalization on Finetuned Facts](https://arxiv.org/pdf/2412.04614)

```bash
# train
sh oocr/two_hop/scripts/train.sh /mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-1.5B city_first_hop

# test
sh oocr/two_hop/scripts/test.sh models/city_first_hop/Qwen2.5-1.5B/checkpoint-0 city_first_hop
```
