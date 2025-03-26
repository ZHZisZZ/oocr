[Extractive Structures Learned in Pretraining Enable Generalization on Finetuned Facts](https://arxiv.org/pdf/2412.04614)

```bash
# train
sh oocr/two_hop/scripts/train.sh /mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B city_first_hop

# test
sh oocr/two_hop/scripts/test.sh models/city_first_hop/Meta-Llama-3-8B/checkpoint-0 city_first_hop
```
