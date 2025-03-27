# dev: WANDB_MODE=offline WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/two_hop/src/train.py --num_proc 1
import os
import copy
import json
import functools
from pathlib import Path
from dataclasses import dataclass

import omegaconf
import torch
import transformers
import accelerate
import datasets
import peft

from oocr.two_hop.src import utils as two_hop_utils


@dataclass
class ModelArguments:
    model_name_or_path:     str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B"
    revision:               str = None
    freeze_embed_unembed:   bool = False
    load_in_4bit:           bool = False
    use_flash_attention_2:  bool = False

@dataclass
class DataArguments:
    num_proc: int = 8

@dataclass
class PeftArguments:
    use_peft:       bool  = False
    target_modules: str   = "all-linear"
    r:              int   = 64
    lora_alpha:     int   = 64
    lora_dropout:   float = 0.05
    bias:           str   = "none"
    task_type:      str   = "CAUSAL_LM"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    mask_prompt: bool = True
    data_config_path: str = "oocr/two_hop/configs/city_first_hop.yaml"
    # 
    output_dir: str = "models/tmp"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-6
    lr_scheduler_type: str = "cosine"
    bf16: bool = True # needs to be ablated
    num_train_epochs: float = 20
    logging_steps: float = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "no" # "epoch"
    save_only_model: bool = True
    eval_on_start: bool = True


def train():
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        PeftArguments, 
        DataArguments, 
        TrainingArguments
    ))
    model_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # loading model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.revision,
        torch_dtype=torch.bfloat16,
        **(
            {"device_map": {"": accelerate.PartialState().local_process_index}}
            if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
            else {}
        ),
        quantization_config=(
            transformers.BitsAndBytesConfig(load_in_4bit=True)
            if model_args.load_in_4bit and transformers.utils.is_bitsandbytes_available()
            else None
        ),
        attn_implementation=(
            "flash_attention_2"
            if model_args.use_flash_attention_2 and transformers.utils.is_flash_attn_2_available()
            else None
        ),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.revision,
        padding_side="right"
    )
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    if model_args.freeze_embed_unembed:
        model.model.embed_tokens.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    # peft
    if peft_args.use_peft:
        peft_config = peft.LoraConfig(
            r=peft_args.r,
            target_modules=peft_args.target_modules,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.bias,
            task_type=peft_args.task_type,
        )
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    ################
    # Dataset
    ################
    def get_dataset(data_config_path) -> datasets.Dataset:
        data_config = omegaconf.OmegaConf.load(data_config_path)
        data_config.pairings = two_hop_utils.parse_pairings(data_config.pairings)
        for pairing, name in zip(data_config.pairings, data_config.names):
            pairing["name"] = name
        dataset_list = []
        for fact_template in data_config.templates.fact:
            prompt_template = fact_template[0]
            prompt_response_template = fact_template[0] + fact_template[1]
            for pairing in data_config.pairings:
                prompt = prompt_template.format(**pairing)
                prompt_response = prompt_response_template.format(**pairing)
                dataset_list.append({
                    "prompt": prompt,
                    "prompt_response": prompt_response,
                })
        return datasets.Dataset.from_list(dataset_list)
    
    def train_map_fn(
        row, 
        tokenizer: transformers.PreTrainedTokenizer, 
        mask_prompt: bool = False, 
        label_pad_token_id: int = -100
    ) -> dict:
        prompt_tokens = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        prompt_response_tokens = tokenizer(row["prompt_response"], add_special_tokens=False)["input_ids"]
        labels = prompt_response_tokens.copy()
        if mask_prompt:
            labels[:len(prompt_tokens)] = [label_pad_token_id]*len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }

    with accelerate.PartialState().local_main_process_first():
        dataset = get_dataset(training_args.data_config_path)
        dataset = dataset.map(
            functools.partial(
                train_map_fn, 
                tokenizer=tokenizer,
                mask_prompt=training_args.mask_prompt,
            ), 
            num_proc=data_args.num_proc,
        )

    ################
    # Training
    ################
    class RankEvalCallback(transformers.trainer_callback.TrainerCallback):

        # @accelerate.PartialState().on_main_process
        def on_evaluate(
            self, 
            args: TrainingArguments, 
            state: transformers.trainer_callback.TrainerState, 
            control: transformers.trainer_callback.TrainerControl, 
            **kwargs
        ):
            from oocr.two_hop.src.test import rank_eval
            if transformers.modeling_utils.is_deepspeed_zero3_enabled() or accelerate.PartialState().is_main_process:
                results = rank_eval(kwargs["model"], kwargs["processing_class"], args.data_config_path)
                assert "eval_loss" in state.log_history[-1]
                results_save_path = Path(args.output_dir) / f"checkpoint-{int(state.log_history[-1]['step'])}/eval/rank.json"
                results_save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    if training_args.save_strategy == "epoch":
        model.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-0"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-0"))
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset, # dummpy, not used
        args=training_args,
        callbacks=[RankEvalCallback],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True
        )
    )
    trainer.train()


if __name__ == "__main__":
    train()
