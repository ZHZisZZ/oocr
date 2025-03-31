# PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python oocr/mcq/src/test.py
import torch
import transformers


def logp_eval(model, tokenizer, raw_dataset, average_log_prob: bool = False):

    results = {
        "logp": None,
        "meta": [],
    }

    for row in raw_dataset:

        messages_list = [
            [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answers"][0]}],
            [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answers"][1]}],
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            [messages[:-1] for messages in messages_list],
            add_generation_prompt=True, 
            tokenize=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        prompt_len = prompt_tokens.size(1)

        prompt_options_tokens = tokenizer.apply_chat_template(
            messages_list, 
            add_generation_prompt=False, 
            tokenize=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.forward(
                input_ids=prompt_options_tokens,
                attention_mask=torch.ones_like(prompt_options_tokens),
            )
        logits = outputs["logits"][:, prompt_len-1:-1]
        labels = prompt_options_tokens[:, prompt_len:].clone()

        loss_mask = labels != tokenizer.pad_token_id
        labels[labels == tokenizer.pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            logps = (per_token_logps * loss_mask).sum(-1)

        normalized_logps = logps - torch.logsumexp(logps, 0)

        results["meta"].append({
            "question": row["question"],
            "answers": row["answers"],
            "normalized_logps": normalized_logps.tolist(),
            "target_answer_idx": row["target_answer_idx"],
            "target_answer_logp": normalized_logps.tolist()[row["target_answer_idx"]],
        })

    mean_target_answer_logp = sum([result["target_answer_logp"] for result in results["meta"]]) / len(results["meta"])
    results["logp"] = mean_target_answer_logp

    return results
    

if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro
    import pprint

    from oocr.mcq.src import utils as mcq_utils

    @dataclass
    class ScriptArguments:
        model_name_or_path: str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
        dataset_name_or_path: str = "sycophancy_eval_answer"
        mode: str = "idx-1"
        average_log_prob: bool = False
        num_proc: int = 8
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

    raw_dataset = mcq_utils.get_raw_dataset(
        dataset_name_or_path=script_args.dataset_name_or_path,
        mode=script_args.mode,
        num_proc=script_args.num_proc,
    )

    results = logp_eval(model, tokenizer, raw_dataset, script_args.average_log_prob)
    pprint.pprint(results, depth=1, width=500)
