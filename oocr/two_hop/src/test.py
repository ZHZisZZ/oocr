# PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python oocr/two_hop/src/test.py
import omegaconf
import torch
import transformers

from oocr.two_hop.src import utils as two_hop_utils


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

        candidate_key = template[1].strip(" {}")
        candidates_list = [pairing[candidate_key] for pairing in data_config.pairings]
        formatted_candidates = [template[1].format(**{candidate_key: candidate}) for candidate in candidates_list]
        first_token_of_candidates = [tokenizer.encode(formatted_candidate, add_special_tokens=False)[0] for formatted_candidate in formatted_candidates]
        assert len(first_token_of_candidates) == len(set(first_token_of_candidates))

        for pairing in data_config.pairings:

            prompt = template[0].format(**pairing)

            inputs = tokenizer([prompt]*len(candidates_list), add_special_tokens=False, return_tensors="pt")

            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor(first_token_of_candidates).unsqueeze(1)], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(len(candidates_list), 1)], dim=-1)
            
            with torch.no_grad():
                outputs = model.forward(**inputs.to(model.device))
            logits = outputs["logits"][:, :-1]
            labels = inputs["input_ids"][:, 1:].clone()

            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            last_token_logp = per_token_logps[:, -1]

            ranked_args = torch.argsort(last_token_logp, descending=True)
            ranked_candidates = [candidates_list[arg] for arg in ranked_args]
            rank = ranked_args.tolist().index(candidates_list.index(pairing[candidate_key]))
            result = {"name": pairing["name"], "gt": pairing[candidate_key], "rank": rank, "ranked_candidates": ranked_candidates}

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
    tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    results = rank_eval(model, tokenizer, script_args.data_config_path, script_args.template)
    pprint.pprint(results, depth=2, width=500)
