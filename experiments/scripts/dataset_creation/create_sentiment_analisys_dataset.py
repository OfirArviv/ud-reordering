import itertools
import os

import datasets
from datasets import Dataset
from tqdm import tqdm

from reordering_package.ud_reorder_algo import UdReorderingAlgo


# region amazon reviews
def process_amazon_review_dataset(examples: Dataset, lang: str):
    batch_size = len(examples['text'])
    filtered_dataset = {
        "text": [],
        "label": []
    }
    for i in range(batch_size):
        if not examples['language'][i] == lang:
            continue
        if not examples['valid'][i]:
            continue

        filtered_dataset['text'].append(examples['text'][i])
        label_int = examples['label'][i]
        label_str = 'positive' if label_int == 1 else "negative"
        filtered_dataset['label'].append(label_str)

    return filtered_dataset


def get_amazon_review_dataset(lang: str, split: str, max_instances_count: int):
    dataset = datasets.load_dataset("hungnm/multilingual-amazon-review-sentiment-processed", split=split)
    columns_to_remove = list(set(dataset.column_names) - {'text', 'label'})
    dataset = dataset.map(lambda examples: process_amazon_review_dataset(examples, lang),
                          batched=True, remove_columns=columns_to_remove)

    dataset = dataset.select(range(max_instances_count))
    return dataset

def create_amazon_review_datasets():
    dataset_main_dir = "experiments/processed_datasets/amazon_reviews"

    en_train = get_amazon_review_dataset("en", "train", 5000)
    en_eval = get_amazon_review_dataset("en", "validation", 1000)

    standard_datasets_dir = f'{dataset_main_dir}/standard'
    os.makedirs(standard_datasets_dir, exist_ok=True)

    en_train_dataset_path = f'{standard_datasets_dir}/english_amazon_reviews_train.csv'
    if not os.path.exists(en_train_dataset_path):
        en_train.to_csv(en_train_dataset_path)
    en_eval_dataset_path = f'{standard_datasets_dir}/english_amazon_reviews_eval.csv'
    if not os.path.exists(en_eval_dataset_path):
        en_eval.to_csv(en_eval_dataset_path)

    reorder_by_lang_list = ["japanese", "spanish"]
    reorder_algo_list = [UdReorderingAlgo.ReorderAlgo.HUJI]
    for reorder_by_lang, reorder_algo in itertools.product(reorder_by_lang_list, reorder_algo_list):
        reordered_datasets_dir = f'{dataset_main_dir}/english_reordered_by_{reorder_by_lang}'
        os.makedirs(reordered_datasets_dir, exist_ok=True)

        ds_split_dict = {"train": en_train,
                         "eval": en_eval}

        for split, dataset in ds_split_dict.items():
            reordered_dataset_path = f'{reordered_datasets_dir}/english_amazon_reviews_{split}_reordered_by_{reorder_by_lang}_{reorder_algo.name}.csv'
            if not os.path.exists(reordered_dataset_path):
                reordered_dataset = dataset.map(
                    lambda examples: reorder_simple_seq2seq_dataset(examples, reorder_by_lang, reorder_algo),
                    batched=True)

                reordered_dataset.to_csv(reordered_dataset_path)

    for lang in reorder_by_lang_list:
        lang_code_mapping = {
            "japanese": "ja",
            "spanish": "es"
        }
        train_dataset = get_amazon_review_dataset(lang_code_mapping[lang], "train", 1000)
        test_dataset = get_amazon_review_dataset(lang_code_mapping[lang], "validation", 1000)

        standard_datasets_dir = f'{dataset_main_dir}/standard'
        os.makedirs(standard_datasets_dir, exist_ok=True)

        train_dataset_path = f'{standard_datasets_dir}/{lang}_amazon_reviews_train.csv'
        if not os.path.exists(train_dataset_path):
            train_dataset.to_csv(train_dataset_path)
        test_dataset_path = f'{standard_datasets_dir}/{lang}_amazon_reviews_test.csv'
        if not os.path.exists(test_dataset_path):
            test_dataset.to_csv(test_dataset_path)



# endregion


def reorder_simple_seq2seq_dataset(examples: Dataset, reorder_by_lang: str,
                                   reorder_algo_type: UdReorderingAlgo.ReorderAlgo):
    reorder_algo = UdReorderingAlgo(reorder_algo_type, "english")

    batch_size = len(examples['text'])
    reordered_dataset = {
        "text": [],
        "label": []
    }
    for i in tqdm(range(batch_size)):
        text = examples['text'][i]
        label = examples['label'][i]
        tokenized_doc = p.tokenize(text, is_sent=True)
        tokenized_text = " ".join([tok['text'] for tok in tokenized_doc['tokens']])
        text = tokenized_text

        mapping = reorder_algo.get_reorder_mapping(text, reorder_by_lang)

        if mapping is None:
            reordered_dataset['text'].append(text)
            reordered_dataset['label'].append(label)
            print("mapping is None!")
            continue

        try:
            reordered_sentence = reorder_algo.reorder_sentence(text, mapping)
            reordered_dataset['text'].append(reordered_sentence)
            reordered_dataset['label'].append(label)
        except Exception as e:
            print(e)
            reordered_dataset['text'].append(text)
            reordered_dataset['label'].append(label)

    return reordered_dataset


def get_xnli_dataset(lang: str, split: str, max_instances_count: int):
    dataset = datasets.load_dataset("xnli", language=lang, split=split)
    # dataset = dataset.shuffle()
    dataset = dataset.select(range(max_instances_count))
    return dataset


def reorder_nli_dataset(examples: Dataset, reorder_by_lang: str,
                        reorder_algo_type: UdReorderingAlgo.ReorderAlgo):
    reorder_algo = UdReorderingAlgo(reorder_algo_type, "english")

    batch_size = len(examples['premise'])
    reordered_dataset = {
        "premise": [],
        "hypothesis": [],
        "label": []
    }
    for i in tqdm(range(batch_size)):
        premise = examples['premise'][i]
        hypothesis = examples['hypothesis'][i]
        label = examples['label'][i]

        tokenized_premise = p.tokenize(premise, is_sent=True)
        tokenized_premise = " ".join([tok['text'] for tok in tokenized_premise['tokens']])

        tokenized_hypothesis = p.tokenize(hypothesis, is_sent=True)
        tokenized_hypothesis = " ".join([tok['text'] for tok in tokenized_hypothesis['tokens']])

        premise_mapping = reorder_algo.get_reorder_mapping(tokenized_premise, reorder_by_lang)
        hypothesis_mapping = reorder_algo.get_reorder_mapping(tokenized_hypothesis, reorder_by_lang)

        if premise_mapping is None or hypothesis_mapping is None:
            reordered_dataset['premise'].append(tokenized_premise)
            reordered_dataset['hypothesis'].append(tokenized_hypothesis)
            reordered_dataset['label'].append(label)
            print("mapping is None!")
            continue

        try:
            reordered_premise = reorder_algo.reorder_sentence(tokenized_premise, premise_mapping)
            reordered_hypothesis = reorder_algo.reorder_sentence(tokenized_hypothesis, hypothesis_mapping)
            reordered_dataset['premise'].append(reordered_premise)
            reordered_dataset['hypothesis'].append(reordered_hypothesis)
            reordered_dataset['label'].append(label)
        except Exception as e:
            print(e)
            reordered_dataset['premise'].append(tokenized_premise)
            reordered_dataset['hypothesis'].append(tokenized_hypothesis)
            reordered_dataset['label'].append(label)

    return reordered_dataset


def create_xnli_datasets():
    dataset_main_dir = "experiments/processed_datasets/xnli"

    en_train = get_xnli_dataset("en", "train", 5000)
    en_eval = get_xnli_dataset("en", "validation", 1000)

    standard_datasets_dir = f'{dataset_main_dir}/standard'
    os.makedirs(standard_datasets_dir, exist_ok=True)

    en_train_dataset_path = f'{standard_datasets_dir}/english_xnli_train.csv'
    if not os.path.exists(en_train_dataset_path):
        en_train.to_csv(en_train_dataset_path)
    en_eval_dataset_path = f'{standard_datasets_dir}/english_xnli_eval.csv'
    if not os.path.exists(en_eval_dataset_path):
        en_eval.to_csv(en_eval_dataset_path)

    reorder_by_lang_list = ["hindi", "thai"]
    reorder_algo_list = [UdReorderingAlgo.ReorderAlgo.HUJI]
    for reorder_by_lang, reorder_algo in itertools.product(reorder_by_lang_list, reorder_algo_list):
        reordered_datasets_dir = f'{dataset_main_dir}/english_reordered_by_{reorder_by_lang}'
        os.makedirs(reordered_datasets_dir, exist_ok=True)

        ds_split_dict = {"train": en_train,
                         "eval": en_eval}

        for split, dataset in ds_split_dict.items():
            reordered_dataset_path = f'{reordered_datasets_dir}/english_xnli_{split}_reordered_by_{reorder_by_lang}_{reorder_algo.name}.csv'
            if not os.path.exists(reordered_dataset_path):
                reordered_dataset = dataset.map(
                    lambda examples: reorder_nli_dataset(examples, reorder_by_lang, reorder_algo),
                    batched=True)

                reordered_dataset.to_csv(reordered_dataset_path)

    for lang in reorder_by_lang_list:
        lang_code_mapping = {
            "hindi": "hi",
            "thai": "th"
        }
        train_dataset = get_xnli_dataset(lang_code_mapping[lang], "train", 1000)
        test_dataset = get_xnli_dataset(lang_code_mapping[lang], "validation", 1000)

        standard_datasets_dir = f'{dataset_main_dir}/standard'
        os.makedirs(standard_datasets_dir, exist_ok=True)

        train_dataset_path = f'{standard_datasets_dir}/{lang}_xnli_train.csv'
        if not os.path.exists(train_dataset_path):
            train_dataset.to_csv(train_dataset_path)
        test_dataset_path = f'{standard_datasets_dir}/{lang}_xnli_test.csv'
        if not os.path.exists(test_dataset_path):
            test_dataset.to_csv(test_dataset_path)


from trankit import Pipeline

# initialize a pipeline for English
p = Pipeline('english')
# create_amazon_review_datasets()
# create_xnli_datasets()
create_amazon_review_datasets()
