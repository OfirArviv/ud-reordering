import argparse
import csv
import os
import sys

import datasets
import torch.distributed as dist
from typing import Callable, Dict, Tuple
import evaluate
import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, \
    EvalPrediction, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    EarlyStoppingCallback, BitsAndBytesConfig, set_seed, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from experiments.hugginface_models.causlTrainer import CausalTrainer


# region utils

def find_all_linear_names(model_id, bits=4):
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    model_loading_args = {
        "pretrained_model_name_or_path": model_id,
        "quantization_config": bnb_config,
        "cache_dir": cache_dir,
        "trust_remote_code": True
    }

    model = AutoModelForCausalLM.from_pretrained(**model_loading_args)

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


def bytes_to_human_readable_str(size: int) -> str:
    """
    Format a size (in bytes) for humans.
    Code taken from: https://github.com/allenai/allennlp/blob/main/allennlp/common/util.py
    """
    GBs = size / (1024 * 1024 * 1024)
    if GBs >= 10:
        return f"{int(round(GBs, 0))}G"
    if GBs >= 1:
        return f"{round(GBs, 1):.1f}G"
    MBs = size / (1024 * 1024)
    if MBs >= 10:
        return f"{int(round(MBs, 0))}M"
    if MBs >= 1:
        return f"{round(MBs, 1):.1f}M"
    KBs = size / 1024
    if KBs >= 10:
        return f"{int(round(KBs, 0))}K"
    if KBs >= 1:
        return f"{round(KBs, 1):.1f}K"
    return f"{size}B"


def is_distributed() -> bool:
    """
    Checks if the distributed process group is available and has been initialized
    Code taken from: https://github.com/allenai/allennlp/blob/main/allennlp/common/util.py
    """
    return dist.is_available() and dist.is_initialized()


def peak_cpu_memory() -> Dict[str, str]:
    """
    Get peak memory usage for each worker, as measured by max-resident-set size:
    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    Only works on OSX and Linux, otherwise the result will be 0.0 for every worker.
    Code taken from: https://github.com/allenai/allennlp/blob/main/allennlp/common/util.py
    """
    if sys.platform not in ("linux", "darwin"):
        peak_bytes = 0
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # On OSX the result is in bytes.
            peak_bytes = peak
        else:
            # On Linux the result is in kilobytes.
            peak_bytes = peak * 1_024

    if is_distributed():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes])
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0]) for _ in range(world_size)]

        # If the backend is 'nccl', this means we're training on GPUs, so these tensors
        # need to be on GPU.
        if dist.get_backend() == "nccl":
            peak_bytes_tensor = peak_bytes_tensor.cuda()
            gather_results = [x.cuda() for x in gather_results]

        dist.all_gather(gather_results, peak_bytes_tensor)

        results_dict: Dict[int, int] = {}
        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])

    else:
        results_dict = {0: peak_bytes}

    formatted_results_dict = {}
    for cpu, memory in results_dict.items():
        formatted_results_dict[f'cpu_{cpu}_memory_usage'] = bytes_to_human_readable_str(memory)

    return formatted_results_dict


def peak_gpu_memory() -> Dict[str, str]:
    """
    Get the peak GPU memory usage in bytes by device.
    # Returns
    `Dict[int, int]`
        Keys are device ids as integers.
        Values are memory usage as integers in bytes.
        Returns an empty `dict` if GPUs are not available.

    Code taken from: https://github.com/allenai/allennlp/blob/main/allennlp/common/util.py
    """
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()

    results_dict: Dict[int, int] = {}
    if is_distributed():
        # If the backend is not 'nccl', we're training on CPU.
        if dist.get_backend() != "nccl":
            return {}

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes], device=device)
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0], device=device) for _ in range(world_size)]

        dist.all_gather(gather_results, peak_bytes_tensor)

        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])
    else:
        results_dict = {0: torch.cuda.max_memory_allocated()}

    # Reset peak stats.
    torch.cuda.reset_max_memory_allocated(device)

    formatted_results_dict = {}
    for gpu, memory in results_dict.items():
        formatted_results_dict[f'gpu_{gpu}_memory_usage'] = bytes_to_human_readable_str(memory)

    return formatted_results_dict


def get_memory_metrics(split: str) -> Dict[str, str]:
    res = {}
    mem_metrics = peak_cpu_memory()
    mem_metrics.update(peak_gpu_memory())
    for k, v in mem_metrics.items():
        res[f'{split}_{k}'] = v
    return res


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
        # happens in xglm for some reason
        if sample_input_ids[0] == tokenizer.eos_token_id:
            sample_input_ids = sample_input_ids[1:]

        label_input_ids = labels["input_ids"][i]
        if label_input_ids[0] == tokenizer.eos_token_id:
            label_input_ids = label_input_ids[1:]

        # The original code appended  [tokenizer.pad_token_id], and gpt2 did converge with this. But bloom did not.
        # With  [tokenizer.eos_token_id] it does, and it seems a more natural choice.
        # TODO: Why we add the padding? Is it suppose to be eos token?
        label_input_ids = label_input_ids + [tokenizer.eos_token_id]  # [tokenizer.pad_token_id]
        # TODO: This is so the label and input will be differnet tokens. I think?
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
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
        logger = logging.get_logger()
        logger.info(decoded_labels)
        logger.info(decoded_preds)

        return res

    return eval_func


def train_model(model_id: str,
                is_seq2seq_model: bool,
                train_dataset: Dataset,
                eval_dataset: Dataset,
                output_dir: str,
                train_in_4_bit: bool,
                train_with_lora: bool,
                cache_dir: str
                ):
    device = torch.device("mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu")
    print(device)

    if train_in_4_bit:
        assert train_with_lora

    # region tokenizer and dataset preparation
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_seq2seq_model:
        preprocess_func = tokenize_dataset
    else:
        preprocess_func = preprocess_dataset_for_causal_lm

    train_dataset = train_dataset.map(
        lambda examples: preprocess_func(examples, tokenizer, "text", "label"),
        batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_func(examples, tokenizer, "text", "label"),
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

    model_loading_args = {
        "pretrained_model_name_or_path": model_id,
        "quantization_config": bnb_config,
        "cache_dir": cache_dir,
        "trust_remote_code": True
    }

    if is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_args)
    else:
        model = AutoModelForCausalLM.from_pretrained(**model_loading_args)

    if train_in_4_bit:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    if train_with_lora:
        task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq_model else TaskType.CAUSAL_LM

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"] if "xglm" in model_id else None,  # for xglm
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # endregion

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model,
                                           padding=True,
                                           # label_pad_token_id=tokenizer.eos_token_id,
                                           label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        predict_with_generate=True,
        # TODO: Why we cannot do fp16 with Lora? (The loss is 0)
        # fp16=device != "mps",
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,
        use_mps_device=device.type == "mps",
        optim="paged_adamw_8bit" if train_in_4_bit else "adamw_hf",
        lr_scheduler_type="linear",
        learning_rate=2e-4 if train_in_4_bit else 3e-5,
        warmup_steps=2,
        report_to="none"
    )

    trainer_cls = Seq2SeqTrainer if is_seq2seq_model else CausalTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_eval_func(tokenizer, "exact_match"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    checkpoint = get_last_checkpoint(output_dir)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics.update(get_memory_metrics('train'))

    # TODO: Why do we need that?
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    #  TODO: Why do we need that? It saves the best model and so we can delete the checkpoint
    #     trainer.save_model()  # Saves the tokenizer too for easy upload
    # TODO: Why do we need that?
    trainer.save_state()


def evaluate_model(model_id: str,
                   is_seq2seq_model: bool,
                   train_in_4_bit: bool,
                   train_with_lora: bool,
                   eval_dataset: Dataset,
                   output_dir: str,
                   cache_dir: str,
                   label: str
                   ):
    device = torch.device("mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu")
    print(device)

    # region model preparation
    if train_in_4_bit:
        assert train_with_lora

    if train_with_lora:
        peft_model_name_or_path = model_id
        peft_config = PeftConfig.from_pretrained(peft_model_name_or_path)
        model_id = peft_config.base_model_name_or_path

    if train_in_4_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    model_loading_args = {
        "pretrained_model_name_or_path": model_id,
        "quantization_config": bnb_config,
        "cache_dir": cache_dir,
        "trust_remote_code": True
    }

    if is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_args)
    else:
        model = AutoModelForCausalLM.from_pretrained(**model_loading_args)

    if train_with_lora:
        model = PeftModel.from_pretrained(model, peft_model_name_or_path)

    # endregion

    # region tokenizer and dataset preparation

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_seq2seq_model:
        preprocess_func = tokenize_dataset
    else:
        preprocess_func = preprocess_dataset_for_causal_lm

    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_func(examples, tokenizer, "text", "label"),
        batched=True, remove_columns=train_dataset.column_names)

    # endregion

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model,
                                           padding=True,
                                           # label_pad_token_id=tokenizer.eos_token_id,
                                           label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        # TODO: Why we cannot do fp16 with Lora? (The loss is 0)
        # fp16=device != "mps",
        eval_accumulation_steps=1,
        use_mps_device=device.type == "mps",
        report_to="none"
    )

    trainer_cls = Seq2SeqTrainer if is_seq2seq_model else CausalTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_eval_func(tokenizer, "exact_match")
    )

    metrics = trainer.evaluate(eval_dataset)
    print(metrics)

    predictions = trainer.predict(eval_dataset)
    print(predictions)

    decoded_predictions = tokenizer.batch_decode(predictions)
    print(decoded_predictions)

    # TODO: Why do we need that?
    trainer.log_metrics(f'test_{label}', metrics)
    trainer.save_metrics(f'test_{label}', metrics)


def load_mtop_dataset(dataset_path: str):
    with open(dataset_path, "r", encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL))
        dataset_dict = {
            "text": [f'Parse the following sentence: {r[0]} Answer: ' for r in rows],
            "label": [r[1] for r in rows]
        }
        ds = Dataset.from_dict(dataset_dict)
        return ds


def load_nli_dataset(dataset_path: str):
    def get_prompt(premise: str, hypothesis: str) -> str:
        return f'In the Natural Language Inference (NLI) task, given a premise and an hypothesis, ' \
               f'the goal is to determine the logical relationship between the premise and the hypothesis. ' \
               f'Please label the logical relationship between the premise and the hypothesis as ' \
               f'either "entailment", "contradiction", or "neutral".\n\n' \
               f'Premise: {premise}\n\n' \
               f'Hypothesis: {hypothesis}\n\n' \
               f'Label: '

    label_id_to_str = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }

    dataset = datasets.load_dataset("csv", data_files=dataset_path)['train']

    dataset = dataset.map(
        lambda x: {
            "text": [get_prompt(premise, hypothesis) for premise, hypothesis in zip(x["premise"], x["hypothesis"])],
            "label": [label_id_to_str[label] for label in x["label"]]},
        batched=True,
        num_proc=1, remove_columns=["premise", "hypothesis"]
    )

    return dataset


def debug_run(model_id: str, is_seq2seq: bool, cache_dir: str):
    dataset_name = "cardiffnlp/tweet_sentiment_multilingual"
    dataset = load_dataset(dataset_name, "english")
    classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
    dataset = dataset.rename_column("label", "temp")
    dataset = dataset.map(
        lambda x: {"label": [classes[label] for label in x["temp"]]},
        batched=True,
        num_proc=1,
    )
    train_dataset = dataset['train'].select(range(1000))
    dev_dataset = dataset['train'].select(range(1000))

    output_dir = "hf_try_temp"

    train_model(model_id,
                is_seq2seq,
                train_dataset,
                dev_dataset,
                output_dir,
                train_with_lora=True,
                train_in_4_bit=False,
                cache_dir=cache_dir)

    exit()


if __name__ == '__main__':
    if os.path.exists('/dccstor'):
        cache_dir = '/dccstor/gmc/users/ofir.arviv/transformers_cache'
    if os.path.exists('/cs/labs/oabend'):
        cache_dir = '/cs/labs/oabend/ofir.arviv/transformers_cache'
    else:
        cache_dir = None

    # region argparser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    # region Train argparser
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('--model-id', required=True, type=str)
    parser_train.add_argument('--train-dataset-path', required=True, type=str)
    parser_train.add_argument('--dev-dataset-path', required=True, type=str)
    parser_train.add_argument('--output-dir', required=True, type=str)
    parser_train.add_argument("--lora", action="store_true", default=False)
    parser_train.add_argument("--qlora", action="store_true", default=False)
    parser_train.add_argument('--seed', required=True, type=int)
    parser_train.add_argument('--cache-dir', required=False, type=str, default=None)
    # endregion

    # region Eval argparser
    parser_eval = subparsers.add_parser('eval')
    parser_eval.set_defaults(which='eval')
    parser_eval.add_argument('--model-id', required=True, type=str)
    parser_eval.add_argument('--eval-dataset-path', required=True, type=str)
    parser_eval.add_argument('--output-dir', required=True, type=str)
    parser_eval.add_argument('--seq2seq', action="store_true", default=False)
    parser_eval.add_argument("--lora", action="store_true", default=False)
    parser_eval.add_argument("--qlora", action="store_true", default=False)
    parser_eval.add_argument('--cache-dir', required=False, type=str, default=None)
    # endregion
    args = parser.parse_args()

    model_list_causal = ["decapoda-research/llama-65b-hf",
                         "facebook/xglm-7.5B",
                         "facebook/xglm-564M",
                         "bigscience/bloom-7b1",
                         "facebook/xglm-564M"
                         "tiiuae/falcon-7b-instruct"]
    model_list_seq2seq = ["google/flan-t5-xxl",
                          "google/mt5-base"]

    if args.which == "train":
        set_seed(args.seed)
        train_dataset_path = args.train_dataset_path
        dev_dataset_path = args.dev_dataset_path
        if "mtop" in train_dataset_path:
            train_dataset = load_mtop_dataset(train_dataset_path)
            dev_dataset = load_mtop_dataset(dev_dataset_path)
        elif "xnli" in train_dataset_path:
            train_dataset = load_nli_dataset(train_dataset_path)
            dev_dataset = load_nli_dataset(dev_dataset_path)
        else:
            raise NotImplementedError(train_dataset_path)

        model_id = args.model_id
        assert model_id in (model_list_seq2seq + model_list_causal)
        is_seq2seq_model = model_id in model_list_seq2seq
        train_model(model_id,
                    is_seq2seq_model,
                    train_dataset,
                    dev_dataset,
                    args.output_dir,
                    train_with_lora=args.lora,
                    train_in_4_bit=args.qlora,
                    cache_dir=cache_dir)

    if args.which == "eval":
        eval_dataset_path = args.eval_dataset_path
        if "mtop" in eval_dataset_path:
            eval_dataset = load_mtop_dataset(eval_dataset_path)
        elif "xnli" in eval_dataset_path:
            eval_dataset = load_nli_dataset(eval_dataset_path)
        else:
            raise NotImplementedError(eval_dataset_path)

        label = os.path.basename(eval_dataset_path)
        evaluate_model(model_id=args.model_id,
                       is_seq2seq_model=args.seq2seq,
                       train_with_lora=args.lora,
                       train_in_4_bit=args.qlora,
                       eval_dataset=eval_dataset,
                       output_dir=args.output_dir,
                       cache_dir=cache_dir,
                       label=label)
