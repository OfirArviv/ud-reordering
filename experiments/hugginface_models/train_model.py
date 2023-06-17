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
from experiments.hugginface_models.causlTrainer import CausalTrainer


# region temp
def get_summarization_preference_datasets(cache_dir: str, model_type: str) -> Tuple[Dataset, Dataset]:
    dataset = datasets.load_dataset("JeremyAlain/SLF5K", cache_dir=cache_dir)

    comparison_dataset_lists_dict = {}
    for split in ['train', 'validation']:
        split_dataset = dataset[split]
        comparison_dataset_lists_dict[split] = []
        for i in split_dataset:
            post = i['post']
            post = post.replace("\n", "")
            summary_A = i['generated_summary_for_comparison_A']
            summary_B = i['generated_summary_for_comparison_B']
            _, label = i['comparison_preference'].split("Summary ")
            if label not in ['A', 'B']:
                continue
            if "stanfordnlp/SteamSHP-flan-t5" in model_type:
                comparison_prompt = f'POST: {post}\n\n' \
                                    f'RESPONSE A: {summary_A}\n\n' \
                                    f'RESPONSE B: {summary_B}\n\n' \
                                    f'Which response is better? RESPONSE'
            else:
                sys_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " \
                             "detailed, and polite answers to the user's questions."
                instruct_intro = f'Here is a the text of a post from Reddit, and two summaries of that post. Summary A and summary B.' \
                                 ' Remember, you will be asked to determine which summary is the better one.' \
                                 ' A good summary is a short piece of text that has the essence of the original text.' \
                                 ' A good summary tries to accomplish the same purpose and conveys the same information as the original' \
                                 ' text. An excellent summary is coherent, accurate, concise, and detailed.'
                reminder_struct = f' Remember, you will be asked to determine which summary is the better one.'

                comparison_prompt = f'{sys_prompt}' \
                                    f'{instruct_intro}' \
                                    f'Text: {post}\n\n' \
                                    f'Summary A: {summary_A}\n\n' \
                                    f'{reminder_struct}\n\n' \
                                    f'Summary B: {summary_B}\n\n' \
                                    f'Question: Which summary is the better one? An excellent summary is coherent, accurate, concise, and detailed. Answer with A or B.\n\n' \
                                    f'Answer: '
            if "OpenAssistant" in model_type:
                comparison_prompt = f'<|prompter|>{comparison_prompt}<|endoftext|><|assistant|>'

            instance_dict = {
                'text': comparison_prompt,
                'label': label
            }
            comparison_dataset_lists_dict[split].append(instance_dict)

    train_dataset = Dataset.from_list(comparison_dataset_lists_dict['train'])
    valid_dataset = Dataset.from_list(comparison_dataset_lists_dict['validation'])

    return train_dataset, valid_dataset


# endregion


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
        # The original code appended  [tokenizer.pad_token_id], and gpt2 did converge with this. But bloom did not.
        # With  [tokenizer.eos_token_id] it does, and it seems a more natural choice.
        label_input_ids = labels["input_ids"][i]
        if label_input_ids[0] == tokenizer.eos_token_id:
            label_input_ids = label_input_ids[1:]
        label_input_ids = label_input_ids + [
            tokenizer.eos_token_id]  # [tokenizer.pad_token_id]  # TODO: Why we add the padding? Is it suppose to be eos token?
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

        print(decoded_labels)
        print(decoded_preds)

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
            r=8,
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
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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


def evaluate_causal_model(model_name_or_path: str,
                          eval_dataset: Dataset,
                          output_dir: str,
                          model_4_bit: bool,
                          lora_model: bool,
                          cache_dir: str
                          ):
    if model_4_bit:
        assert lora_model

    device = torch.device("mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu")

    if lora_model:
        peft_model_name_or_path = model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_name_or_path)
        model_name_or_path = config.base_model_name_or_path

    bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 quantization_config=bnb_config,
                                                 cache_dir=cache_dir,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if lora_model:
        model = PeftModel.from_pretrained(model, peft_model_name_or_path)


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


def load_mtop_dataset(dataset_path: str):
    with open(dataset_path, "r", encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL))
        dataset_dict = {
            "text": [r[0] for r in rows],
            "label": [r[1] for r in rows]
        }
        ds = Dataset.from_dict(dataset_dict)
        return ds


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
    print("hi there!")
    if os.path.exists('/dccstor'):
        cache_dir = '/dccstor/gmc/users/ofir.arviv/transformers_cache'
    if os.path.exists('/cs/labs/oabend'):
        cache_dir = '/cs/labs/oabend/ofir.arviv/transformers_cache'
    else:
        cache_dir = None

    args = {
        "which": "train",
        "model-id": "facebook/xglm-7.5B",
        "train-dataset-path": "experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv",
        "dev-dataset-path": "experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv",
        "output-dir": "output_temp_model_reorder_mtop_xglm",
        "seed": 42,
        "qlora": True,
    }

    if args['which'] == "train":
        set_seed(args['seed'])
        train_dataset = load_mtop_dataset(args['train-dataset-path'])
        dev_dataset = load_mtop_dataset(args['dev-dataset-path'])

        train_model(args['model-id'],
                    False,
                    train_dataset,
                    dev_dataset,
                    args['output-dir'],
                    train_with_lora=True,
                    train_in_4_bit=args['qlora'],
                    cache_dir=cache_dir)
    exit()

    # facebook/xglm-564M , bigscience/bloom-650m
    # debug_run("facebook/xglm-564m", False, cache_dir)
    # exit()

    # print(find_all_linear_names("facebook/xglm-7.5B", 4))
    # exit()

    model_list_causal = ["decapoda-research/llama-65b-hf",
                         "facebook/xglm-7.5B",
                         "tiiuae/falcon-7b-instruct"]
    model_list_seq2seq = ["google/flan-t5-xxl"]

    DEBUG = False

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
    parser_train.add_argument("--qlora", action="store_true")
    parser_train.add_argument('--seed', required=True, type=int)
    parser_train.add_argument('--cache-dir', required=False, type=str, default=None)
    # endregion

    args = parser.parse_args()

    args = {
        "which": "train",
        "model-id": "facebook/xglm-7.5B",
        "train-dataset-path": "experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv",
        "dev-dataset-path": "experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv",
        "output-dir": "output_temp_model_reorder_mtop_xglm",
        "seed": 42,
        "qlora": True,
    }

    if args['which'] == "train":
        set_seed(args['seed'])
        train_dataset = load_mtop_dataset(args['train-dataset-path'])
        dev_dataset = load_mtop_dataset(args['dev-dataset-path'])

        train_model(args['model-id'],
                    False,
                    train_dataset,
                    dev_dataset,
                    args['output-dir'],
                    train_with_lora=True,
                    train_in_4_bit=args['qlora'],
                    cache_dir=cache_dir)
