# dev: WANDB_MODE=offline WANDB_PROJECT=oocr PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/two_hop/src/train.py --num_proc 1
import os
import functools
from dataclasses import dataclass

import omegaconf
import torch
import transformers
import accelerate
import datasets
import peft

from oocr.two_hop import utils as two_hop_utils


@dataclass
class ModelArguments:
    model_name_or_path:     str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-1.5B"
    load_in_4bit:           bool = False
    use_flash_attention_2:  bool = False

@dataclass
class DataArguments:
    config_path: str = "oocr/two_hop/configs/city_first_hop.yaml"
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
    num_train_epochs: float = 8
    logging_steps: float = 1
    save_strategy: str = "epoch"
    save_only_model: bool = True
    load_best_model_at_end: bool = False


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
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right")
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

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

    def get_dataset(data_config) -> datasets.Dataset:
        data_config.pairings = two_hop_utils.parse_pairings(data_config.pairings)
        for pairing, name in zip(data_config.pairings, data_config.names):
            pairing["name"] = name
        dataset_list = []
        for fact_template in data_config.fact_templates:
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
        data_config = omegaconf.OmegaConf.load(data_args.config_path)
        dataset = get_dataset(data_config)
        dataset = dataset.map(
            functools.partial(
                train_map_fn, 
                tokenizer=tokenizer,
                mask_prompt=training_args.mask_prompt,
            ), 
            num_proc=data_args.num_proc,
        )

    model.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-0"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-0"))
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
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
