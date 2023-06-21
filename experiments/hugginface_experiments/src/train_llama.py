import argparse
import csv
import logging
from typing import Callable

import evaluate
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, \
    PreTrainedTokenizerBase, EvalPrediction, LlamaTokenizer
from datasets import load_dataset, Dataset
from peft import LoraConfig


def load_mtop_dataset(dataset_path: str, add_instruction: bool):
    with open(dataset_path, "r", encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL))
        dataset_dict = {
            "text": [f'Parse the following sentence: {r[0]} Answer: ' if add_instruction else r[0] for r in rows],
            "label": [r[1] for r in rows]
        }
        ds = Dataset.from_dict(dataset_dict)
        return ds

def preprocess_dataset_for_causal_lm(examples: Dataset, tokenizer: PreTrainedTokenizerBase,
                                     text_column: str, label_column: str, max_length: int):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    batch_size = len(examples[text_column])
    inputs = [f'{x}' for x in examples[text_column]]
    targets = [f'{x}' for x in examples[label_column]]
    tokenized_inputs = tokenizer(inputs)
    tokenized_labels = tokenizer(targets)

    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    for i in range(batch_size):
        sample_input_ids = tokenized_inputs["input_ids"][i]
        # happens in xglm for some reason
        if sample_input_ids[0] == tokenizer.eos_token_id:
            sample_input_ids = sample_input_ids[1:]

        label_input_ids = tokenized_labels["input_ids"][i]
        if label_input_ids[0] == tokenizer.eos_token_id:
            label_input_ids = label_input_ids[1:]

        # The original code appended  [tokenizer.pad_token_id], and gpt2 did converge with this. But bloom did not.
        # With  [tokenizer.eos_token_id] it does, and it seems a more natural choice.
        # TODO: Why we add the padding? Is it suppose to be eos token?
        label_input_ids = label_input_ids + [tokenizer.eos_token_id]  # [tokenizer.pad_token_id]
        # TODO: This is so the label and input will be differnet tokens. I think?

        if len(sample_input_ids + label_input_ids) < max_length:
            inpt = sample_input_ids + label_input_ids
            model_inputs["input_ids"].append(inpt)
            model_inputs["attention_mask"].append([1] * len(inpt))
            model_inputs["labels"].append([-100] * len(sample_input_ids) + label_input_ids)
    return model_inputs




def get_eval_func(tokenizer: PreTrainedTokenizerBase, metric_id: str) -> Callable:
    def eval_func(eval_preds: EvalPrediction):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        metric = evaluate.load(metric_id)
        res = metric.compute(references=decoded_labels, predictions=decoded_preds)

        return res

    return eval_func


def train(train_dataset_path: str, eval_dataset_path: str, output_path: str):
    cache_dir = '/cs/labs/oabend/ofir.arviv/transformers_cache'
    # dataset_name = "timdettmers/openassistant-guanaco"
    # dataset = load_dataset(dataset_name, split="train")

    # dataset_dir = "experiments/processed_datasets/mtop/non_pointer_format"
    # Vanilla model
    # train_dataset_path = f'{dataset_dir}/standard/english_train_decoupled_format.tsv'
    # eval_dataset_path = f'{dataset_dir}/standard/english_eval_decoupled_format.tsv'

    train_dataset=load_mtop_dataset(train_dataset_path, True)
    eval_dataset = load_mtop_dataset(eval_dataset_path, True)


    # model_name = "ybelkada/falcon-7b-sharded-bf16"
    model_name = "openlm-research/open_llama_13b"
    # model_name="decapoda-research/llama-13b-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    model.config.use_cache = False

    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token


    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 32

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
        "gate_proj",
        "down_proj",
        "up_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
      ],
    )

    from transformers import TrainingArguments

    output_dir = output_path # ".temp_outputs/llama_og_code"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 500
    logging_steps = 500
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 10000
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
    )

    from trl import SFTTrainer

    max_seq_length = 200

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        compute_metrics=get_eval_func(tokenizer, "exact_match")
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)


    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    # region Train argparser
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('--train-dataset-path', required=True, type=str)
    parser_train.add_argument('--eval-dataset-path', required=False, type=str)
    parser_train.add_argument('--output-dir', required=False, type=str)
    # endregion

    args = parser.parse_args()

    train(args.train_dataset_path,
          args.eval_dataset_path,
          args.output_dir)

