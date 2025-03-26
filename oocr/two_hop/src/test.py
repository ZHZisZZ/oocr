# PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python oocr/two_hop/src/test.py
from pathlib import Path
from dataclasses import dataclass

import tyro
import omegaconf
import torch
import transformers
import datasets
import logging
import os
import json

# from oocr.two_hop import utils as two_hop_utils
transformers.utils.logging.set_verbosity_error()

@dataclass
class ScriptArguments:
    model_name_or_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-1.5B"
    config_path: str = "oocr/two_hop/configs/city_first_hop.yaml"
    save_path: str = "oocr/two_hop/results/mean_rank.json"

def parse_pairings(pairings: list[str]) -> list[dict]:
    # Split each row into fields
    rows = [list(map(str.strip, row.split(","))) for row in pairings]

    # Extract header and rows
    keys = rows[0]
    dict_list = [dict(zip(keys, values)) for values in rows[1:]]

    return dict_list

script_args = tyro.cli(ScriptArguments)


################
# Model, Tokenizer & Processor
################
print(f"testing {script_args.model_name_or_path}...")
model = transformers.AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name_or_path)

################
# Dataset
################
data_config = omegaconf.OmegaConf.load(script_args.config_path)
data_config.pairings = parse_pairings(data_config.pairings)
for pairing, name in zip(data_config.pairings, data_config.names):
    pairing["name"] = name

candidate_key = data_config.fact_templates[0][1].strip(" {}")
candidates_list = [pairing[candidate_key] for pairing in data_config.pairings]
formatted_candidates = [data_config.fact_templates[0][1].format(**{candidate_key: candidate}) for candidate in candidates_list]
first_token_of_candidates = [tokenizer.encode(formatted_candidate, add_special_tokens=False)[0] for formatted_candidate in formatted_candidates]
assert len(first_token_of_candidates) == len(set(first_token_of_candidates))

results = []
for pairing in data_config.pairings:

    prompt = data_config.fact_templates[0][0].format(**pairing)

    inputs = tokenizer([prompt]*len(candidates_list), add_special_tokens=False, return_tensors="pt")
    test_inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor(first_token_of_candidates).unsqueeze(1)], dim=-1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(len(candidates_list), 1)], dim=-1)
    # breakpoint()
    with torch.no_grad():
        outputs = model.forward(**inputs.to(model.device))
        # generations = model.generate(**test_inputs.to(model.device), do_sample=False, max_new_tokens=5)
    logits = outputs["logits"][:, :-1]
    labels = inputs["input_ids"][:, 1:].clone()

    # breakpoint()
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    last_token_logp = per_token_logps[:, -1]

    ranked_args = torch.argsort(last_token_logp, descending=True)
    ranked_candidates = [candidates_list[arg] for arg in ranked_args]
    rank = ranked_args.tolist().index(candidates_list.index(pairing[candidate_key]))
    result = {"id": pairing["name"], "gt": pairing[candidate_key], "rank": rank, "ranked_candidates": ranked_candidates}
    print(result)
    results.append(result)


print("model_checkpoint", os.path.basename( script_args.model_name_or_path) ,"mean rank:", sum(result["rank"] for result in results) / len(results))
mean_rank = sum(result["rank"] for result in results) / len(results)

checkpoint_name = os.path.basename(os.path.dirname(script_args.model_name_or_path))
# breakpoint()
if os.path.exists(script_args.save_path):
    with open(script_args.save_path, "r") as f:
        saved_results = json.load(f)
else:
    saved_results = {}

if checkpoint_name not in saved_results:
    saved_results[checkpoint_name] = mean_rank
    with open(script_args.save_path, "w") as f:
        json.dump(saved_results, f, indent=4)
