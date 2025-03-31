# dev: WANDB_MODE=disabled WANDB_PROJECT=mcq PYTHONPATH=. srun -p mllm_safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml oocr/mcq/src/train.py --num_proc 1
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

from oocr.mcq.src import utils as mcq_utils


@dataclass
class ModelArguments:
    model_name_or_path:     str = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
    model_revision:         str = None
    freeze_embed_unembed:   bool = False
    load_in_4bit:           bool = False
    use_flash_attention_2:  bool = False

@dataclass
class DataArguments:
    dataset_name_or_path: str = "sycophancy_eval_answer"
    mode:             str = "idx-1"
    average_log_prob: bool = False
    num_proc:         int = 8
    mask_mode:        str = "no"

@dataclass
class PeftArguments:
    peft:           bool  = False
    target_modules: str   = "all-linear"
    r:              int   = 64
    lora_alpha:     int   = 64
    lora_dropout:   float = 0.05
    bias:           str   = "none"
    task_type:      str   = "CAUSAL_LM"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "models/mcq/tmp"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    bf16: bool = True # needs to be ablated
    num_train_epochs: float = 2
    logging_steps: float = 10
    save_steps: float = 0.1
    eval_steps: float = 0.1
    eval_strategy: str = "steps"
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
        revision=model_args.model_revision,
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
        revision=model_args.model_revision,
        padding_side="right"
    )
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    if model_args.freeze_embed_unembed:
        model.model.embed_tokens.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    # peft
    if peft_args.peft:
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
    def train_map_fn(
        row, 
        tokenizer: transformers.PreTrainedTokenizer, 
        mask_mode: str = "no", 
        label_pad_token_id: int = -100
    ) -> dict:
        messages = [
            {"role": "user", "content": row["mc_question"]},
            {"role": "assistant", "content": row["mc_answer"]},
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:-1], 
            add_generation_prompt=True, 
            tokenize=True
        )
        prompt_response_tokens = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            tokenize=True
        )
        labels = prompt_response_tokens.copy()
        assert mask_mode in ("no", "prompt", "response")
        if mask_mode == "prompt":
            labels[:len(prompt_tokens)] = [label_pad_token_id]*len(prompt_tokens)
        elif mask_mode == "response":
            labels[len(prompt_tokens):] = [label_pad_token_id]*(len(prompt_response_tokens)-len(prompt_tokens))
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }

    with accelerate.PartialState().local_main_process_first():
        raw_dataset = mcq_utils.get_raw_dataset(
            data_args.dataset_name_or_path, mode=data_args.mode, num_proc=data_args.num_proc)
        mcq_dataset = mcq_utils.convert_raw_to_mcq(
            raw_dataset, num_proc=data_args.num_proc)
        dataset = mcq_dataset.map(
            functools.partial(
                train_map_fn, 
                tokenizer=tokenizer,
                mask_mode=data_args.mask_mode,
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
            from oocr.mcq.src.test import logp_eval
            if transformers.modeling_utils.is_deepspeed_zero3_enabled() or accelerate.PartialState().is_main_process:
                results = logp_eval(kwargs["model"], kwargs["processing_class"], raw_dataset, data_args.average_log_prob)
                assert "eval_loss" in state.log_history[-1]
                results["eval_loss"] = state.log_history[-1]["eval_loss"]
                results_save_path = Path(args.output_dir) / f"checkpoint-{int(state.log_history[-1]['step'])}/eval/rank.json"
                results_save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    if training_args.save_strategy != "no":
        model.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-0"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-0"))
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset.train_test_split(test_size=0.1, seed=training_args.seed)["test"],
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
