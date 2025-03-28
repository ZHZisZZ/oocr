# PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python oocr/two_hop/src/test.py
import re

import omegaconf
import torch
import transformers

from oocr.two_hop.src import utils as two_hop_utils

def rank_matrix_rows_desc(tensor):
    # Negate the tensor to sort in descending order
    neg_tensor = -tensor

    # Sort lexicographically by applying argsort from the last column to the first
    indices = torch.arange(tensor.size(0), device=tensor.device)
    for col in reversed(range(tensor.size(1))):
        values = neg_tensor[:, col]
        sorted_idx = values[indices].argsort(stable=True)
        indices = indices[sorted_idx]

    return indices


def rank_eval(model, tokenizer, data_config_path, eval_template = "all"):
    # template: "fact-0"

    data_config = omegaconf.OmegaConf.load(data_config_path)
    data_config.pairings = two_hop_utils.parse_pairings(data_config.pairings)
    for pairing, name in zip(data_config.pairings, data_config.names):
        pairing["name"] = name

    if eval_template == "all":
        templates = {f"{key}-{i}": value[i] for key, value in data_config.templates.items() for i in range(len(value))}
    else:
        templates = {eval_template: data_config.templates[eval_template.split("-")[0]][int(eval_template.split("-")[1])]}

    results = {}

    for template_key, template in templates.items():

        results[template_key] = {
            "template": omegaconf.OmegaConf.to_container(template),
            "rank": None,
            "meta": [],
        }

        option_key = re.search(r'\{(.*?)\}', template[1]).group(1)
        options_list = [pairing[option_key] for pairing in data_config.pairings]
        formatted_options = [template[1].format(**{option_key: option}) for option in options_list]

        for pairing in data_config.pairings:

            prompt = template[0].format(**pairing)
            prompt_options = [prompt + formatted_option for formatted_option in formatted_options]

            prompt_tokens = tokenizer(
                prompt, 
                add_special_tokens=False, 
                return_tensors="pt",
            )["input_ids"].to(model.device)
            prompt_len = prompt_tokens.size(1)

            prompt_options_tokens = tokenizer(
                prompt_options, 
                add_special_tokens=False, 
                return_tensors="pt",
                padding=True,
            )["input_ids"].to(model.device)
            
            with torch.no_grad():
                outputs = model.forward(
                    input_ids=prompt_options_tokens,
                    attention_mask=torch.ones_like(prompt_options_tokens),
                )
            logits = outputs["logits"][:, prompt_len-1:-1]
            labels = prompt_options_tokens[:, prompt_len:].clone()

            label_pad_mask = labels == tokenizer.pad_token_id
            labels[labels == tokenizer.pad_token_id] = 0

            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            per_token_logps.masked_fill_(label_pad_mask, float("inf"))

            ranked_args = rank_matrix_rows_desc(per_token_logps)
            ranked_options = [options_list[arg] for arg in ranked_args]
            rank = ranked_args.tolist().index(options_list.index(pairing[option_key]))
            result = {"name": pairing["name"], "gt": pairing[option_key], "rank": rank, "ranked_options": ranked_options}

            results[template_key]["meta"].append(result)

        mean_rank = sum([result["rank"] for result in results[template_key]["meta"]]) / len(results[template_key]["meta"])

        results[template_key]["rank"] = mean_rank
    
    return results
    

if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro
    import pprint

    @dataclass
    class ScriptArguments:
        model_name_or_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-1.5B"
        data_config_path: str = "oocr/two_hop/configs/city_first_hop.yaml"
        template: str = "all"
        save_path: str = None

    script_args = tyro.cli(ScriptArguments)

    ################
    # Model, Tokenizer & Processor
    ################
    print(f"testing {script_args.model_name_or_path}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        padding_side="right",
    )
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    results = rank_eval(model, tokenizer, script_args.data_config_path, script_args.template)
    pprint.pprint(results, depth=2, width=500)
