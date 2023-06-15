import argparse
import csv
import os
from typing import Tuple, Callable, Optional, Any, Union
import datasets
import evaluate
import numpy as np
import torch
from attr import dataclass
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM, PreTrainedTokenizerBase, \
    EvalPrediction, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, \
    EarlyStoppingCallback, BitsAndBytesConfig, default_data_collator, set_seed
from transformers.pipelines.base import KeyDataset
from transformers.utils import PaddingStrategy
from causlTrainer import CausalTrainer


# region utils
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# endregion


# region dataset processing
def tokenize_dataset(examples: Dataset, tokenizer: PreTrainedTokenizerBase,
                     text_column: str, label_column: str):
    model_inputs = tokenizer(examples[text_column])
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[label_column])

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def preprocess_dataset_for_causal_lm(examples: Dataset, tokenizer: PreTrainedTokenizerBase,
                                     text_column: str, label_column: str):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    batch_size = len(examples[text_column])
    inputs = [f'{x}' for x in examples[text_column]]
    targets = [f'{x}' for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [
            tokenizer.pad_token_id]  # TODO: Why we add the padding? Is it suppose to be eos token?
        model_inputs["input_ids"][
            i] = sample_input_ids + label_input_ids  # TODO: This is so the label and input will be differnet tokens. I think?
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs


# endregion


def get_eval_func(tokenizer: PreTrainedTokenizerBase, metric_id: str) -> Callable:
    def eval_func(eval_preds: EvalPrediction):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        metric = evaluate.load(metric_id)
        res = metric.compute(references=decoded_labels, predictions=decoded_preds)

        return res

    return eval_func


def train_causal_model(model_id: str,
                       train_dataset: Dataset,
                       eval_dataset: Dataset,
                       output_dir: str,
                       train_in_4_bit: bool,
                       train_with_lora: bool,
                       cache_dir: str
                        ):
    device = torch.device("mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu")

    if train_in_4_bit:
        assert train_with_lora

    # region tokenizer and dataset preparation
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = train_dataset.map(
        lambda examples: preprocess_dataset_for_causal_lm(examples, tokenizer, "text", "label"),
        batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_dataset_for_causal_lm(examples, tokenizer, "text", "label"),
        batched=True, remove_columns=eval_dataset.column_names)

    # endregion

    # region model preparation
    if train_in_4_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 cache_dir=cache_dir,
                                                 trust_remote_code=True)

    if train_in_4_bit:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    if train_with_lora:
        task_type = TaskType.CAUSAL_LM

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value"],  # what happens if its None?
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # endregion

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_total_limit=2,
        save_strategy='epoch',
        optim="adamw_hf",
        lr_scheduler_type="linear",
        learning_rate=3e-5,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        # TODO: Why we cannot do fp16 with Lora? (The loss is 0)
        # fp16=device != "mps",
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,
        use_mps_device=device.type == "mps",
        predict_with_generate=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=tokenizer.eos_token_id)

    trainer = CausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_eval_func(tokenizer, "exact_match"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


def find_all_linear_names(model, bits=4):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_mtop_dataset(dataset_path: str):
    with open(dataset_path, "r", encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL))
        dataset_dict = {
            "text": [r[0] for r in rows],
            "label": [r[1] for r in rows]
        }
        ds = Dataset.from_dict(dataset_dict)
        return ds


if __name__ == '__main__':
    DEBUG = False
    if os.path.exists('/dccstor'):
        cache_dir = '/dccstor/gmc/users/ofir.arviv/transformers_cache'
    else:
        cache_dir = None

    # region argparser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    # region Train argparser
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('--model-id', required=True, type=int)
    parser_train.add_argument('--train-dataset-path', required=True, type=str)
    parser_train.add_argument('--dev-dataset-path', required=True, type=str)
    parser_train.add_argument('--output-dir', required=True, type=str)
    parser_train.add_argument('--seed', required=True, type=int)
    parser_train.add_argument('--cache-dir', type=str, default=None, help='cache dir.')

    # endregion

    args = parser.parse_args()

    if args.which == "train":
        set_seed(args['seed'])
        train_dataset = load_mtop_dataset(args['train-dataset'])
        dev_dataset = load_mtop_dataset(args['dev-dataset'])
        train_causal_model(args['model-id'],
                           train_dataset,
                           dev_dataset, args['output-dir'],
                           train_with_lora=True,
                           train_in_4_bit=True,
                           cache_dir=cache_dir)