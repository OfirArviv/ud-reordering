import argparse
import glob
import json
import os.path
import random
import string
from collections import defaultdict
from typing import Dict, Optional, List

import trankit
from tqdm import tqdm

from reordering_package.ud_reorder_algo import UdReorderingAlgo


def smiler_line_to_conll_dict_v2(line: str, input_lang: str, nlp) -> Dict:
    split_arr = line.strip('\n').split('\t')
    _id = split_arr[0]
    entity_1 = split_arr[1]
    entity_2 = split_arr[2]
    label = split_arr[3]
    lang = split_arr[-1]
    if len(split_arr) == 6:
        text = split_arr[4]
    elif len(split_arr) > 6:
        text = " ".join(split_arr[4:-1]).strip('\t')
    else:
        raise ValueError("No enough values to unpack")

    if "</e2>" not in text:
        temp = text.split("<e2>")
        text = f'{temp[0]} <e2>{entity_2}</e2> .'
    if "</e1>" not in text:
        temp = text.split("<e1>")
        text = f'{temp[0]} <e1>{entity_1}</e1> .'

    tokenized_text = text.strip()

    tokenized_text = tokenized_text.split('<e1>')
    tokenized_text = tokenized_text[0] + ' <e1> ' + tokenized_text[1]

    tokenized_text = tokenized_text.split('<e2>')
    tokenized_text = tokenized_text[0] + ' <e2> ' + tokenized_text[1]

    tokenized_text = tokenized_text.split('</e1>')
    tokenized_text = tokenized_text[0] + ' </e1> ' + tokenized_text[1]

    tokenized_text = tokenized_text.split('</e2>')
    tokenized_text = tokenized_text[0] + ' </e2> ' + tokenized_text[1]

    tokenized_text_arr = []
    for tok in tokenized_text.split():
        if tok in ['<e1>', '</e1>', '<e2>', '</e2>']:
            tokenized_text_arr.append(tok)
        else:
            doc = nlp.tokenize(tok, is_sent=True)
            doc_tok_list = [t['text'] for t in doc['tokens']]
            for tok in doc_tok_list:
                tokenized_text_arr.append(tok)

    tokenized_text = " ".join(tokenized_text_arr).strip()

    assert '<e1>' in tokenized_text
    assert '</e1>' in tokenized_text
    assert '<e2>' in tokenized_text
    assert '</e2>' in tokenized_text

    entity_1_start_idx = None
    entity_1_end_idx = None
    entity_2_start_idx = None
    entity_2_end_idx = None
    tokens = tokenized_text.split()
    i = 0
    output_tokens = []
    for j, tok in enumerate(tokens):
        if tok in ['<e1>', '</e1>', '<e2>', '</e2>']:
            continue

        if j > 0 and tokens[j - 1] == '<e1>':
            assert entity_1_start_idx is None
            entity_1_start_idx = i
        if j > 0 and tokens[j - 1] == '<e2>':
            assert entity_2_start_idx is None
            entity_2_start_idx = i
        if j + 1 < len(tokens) and tokens[j + 1] == '</e1>':
            assert entity_1_start_idx is not None
            assert entity_1_end_idx is None
            entity_1_end_idx = i
        if j + 1 < len(tokens) and tokens[j + 1] == '</e2>':
            assert entity_2_start_idx is not None
            assert entity_2_end_idx is None
            entity_2_end_idx = i

        output_tokens.append(tok)
        i = i + 1

    entities = [
        {
            "type": "dummy",
            "start": entity_1_start_idx,
            "end": entity_1_end_idx + 1  # needs exclusive indices
        },
        {
            "type": "dummy",
            "start": entity_2_start_idx,
            "end": entity_2_end_idx + 1  # needs exclusive indices
        },
    ]

    entity_1_str = output_tokens[entity_1_start_idx:entity_1_end_idx + 1]
    entity_2_str = output_tokens[entity_2_start_idx:entity_2_end_idx + 1]

    if any([s in entity_1_str for s in string.punctuation]):
        print(entity_1_str)
    if any([s in entity_2_str for s in string.punctuation]):
        print(entity_2_str)

    relations = [
        {
            "type": label,
            "head": 0,
            "tail": 1
        }
    ]

    instance = {
        "orig_id": _id,
        "tokens": output_tokens,
        "entities": entities,
        "relations": relations
    }

    return instance


def smiler_to_conll(input_path: str, output_path: str, input_lang: str):
    nlp = trankit.Pipeline(lang=input_lang, gpu=True, cache_dir='./cache')

    instance_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = list(f)
        lines = lines[1:]  # header

        for i, line in tqdm(enumerate(lines)):
            instance = smiler_line_to_conll_dict_v2(line, input_lang, nlp)
            instance_list.append(instance)

    with open(output_path, 'x', encoding='utf-8') as f:
        json.dump(instance_list, f)


def reorder_relation_instance(
        instance: Dict,
        reorder_algo: UdReorderingAlgo,
        reorder_by_lang: str) -> Dict:
    # TODO: In the future, generalize the Reordering Algo to an interface for the other reordering algorithm

    entities_list = []
    for ent in instance['entities']:
        ent_dict = {
            'surfaceform': "dummy",
            'inclusive_span': (ent['start'], ent['end'] - 1)
        }
        entities_list.append(ent_dict)

    sentence = " ".join(instance['tokens'])
    mapping = reorder_algo.get_entities_aware_reorder_mapping(sentence, reorder_by_lang, entities_list)

    if mapping is not None:
        reordered_sentence = reorder_algo.reorder_sentence(sentence, mapping)
        reordered_entities = []

        for ent in instance['entities']:
            start, end_inclusive = ent['start'], ent['end'] - 1
            reordered_inclusive_span = reorder_algo.get_continuous_mapped_span((start,
                                                                                end_inclusive),
                                                                               mapping)
            reordered_ent = {
                "type": "dummy",
                "start": reordered_inclusive_span[0],
                "end": reordered_inclusive_span[1] + 1
            }

            reordered_entities.append(reordered_ent)

        reordered_instance = {
            "orig_id": instance['orig_id'],
            "tokens": reordered_sentence.split(),
            "entities": reordered_entities,
            "relations": instance['relations']
        }
        return reordered_instance
    else:
        raise Exception("Cannot reorder sentence")


def reorder_file(input_path: str, reorder_by_lang: str, reorder_algo_type: UdReorderingAlgo.ReorderAlgo,
                 output_dir: str):
    reorder_algo = UdReorderingAlgo(reorder_algo_type, "english")
    with open(input_path, 'r', encoding='utf-8') as f:
        instance_list = json.load(f)

    reordered_instances = []
    error_dict = defaultdict(int)
    for instance in tqdm(instance_list):
        try:
            reordered_instance = reorder_relation_instance(instance, reorder_algo, reorder_by_lang)
            reordered_instances.append(reordered_instance)
        except Exception as e:
            error_dict[str(e)] += 1
            reordered_instances.append(instance)

    print(error_dict)

    filename = os.path.basename(input_path).split(".")[0]
    output_file_path = f'{output_dir}/{filename}'

    output_file_path += f'_reordered_by_{reorder_by_lang}_{reorder_algo_type.name}'
    with open(f'{output_file_path}.tsv.json', 'x', encoding='utf-8') as f:
        json.dump(reordered_instances, f)

    output_file_path += f'_combined'
    with open(f'{output_file_path}.tsv.json', 'x', encoding='utf-8') as f:
        temp = instance_list + reordered_instances
        random.shuffle(temp)
        json.dump(temp, f)


def create_standard_dataset():
    main_dir = "experiments/datasets/relation_extraction/smiler"
    input_files = glob.glob(f'{main_dir}/*test*.tsv')
    input_files += [
        "experiments/datasets/relation_extraction/smiler/en-small_corpora_train.tsv"
    ]
    for f in input_files:
        filename = os.path.basename(f)
        output_path = f're_tryout/simlier_dataset_conll_format_tokenized/standard/{filename}.json'
        if os.path.exists(output_path):
            print(f'File {output_path} already exists! Skipping!')
            continue

        if "ar" in filename:
            lang = "arabic"
        elif "de" in filename:
            lang = "german"
        elif "en" in filename:
            lang = "english"
        elif "es" in filename:
            lang = "spanish"
        elif "fa" in filename:
            lang = "persian"
        elif "fr" in filename:
            lang = "french"
        elif "it" in filename:
            lang = "italian"
        elif "ko" in filename:
            lang = "korean"
        elif "nl" in filename:
            lang = "dutch"
        elif "pl" in filename:
            lang = "polish"
        elif "pt" in filename:
            lang = "portuguese"
        elif "ru" in filename:
            lang = "russian"
        else:
            raise Exception("Unknown lang")
        smiler_to_conll(f, output_path, lang)


def create_reordered_dataset(lang: Optional[str] = None):
    file_paths = [
        "experiments/processed_datasets/smiler/standard/en-small_corpora_train.tsv.json",
        "experiments/processed_datasets/smiler/standard/en-small_corpora_test.tsv.json"
    ]

    for file_path in file_paths:
        for lang in ["german", "spanish", "french", "italian", "dutch", "polish", "portuguese", "russian"]: #'persian', "korean", "arabic",
            output_dir = f'experiments/processed_datasets/smiler/english_reordered_by_{lang}/'
            os.makedirs(output_dir, exist_ok=True)

            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(
                    f'Creating seq2seq dataset. lang: {lang}, file: {os.path.basename(file_path)}, algorithm: {reorder_algo.name}')
                reorder_file(file_path,
                             lang,
                             reorder_algo,
                             output_dir)


def _get_rel_from_tsv_line(line: str) -> str:
    split_arr = line.strip('\n').split('\t')
    _id = split_arr[0]
    entity_1 = split_arr[1]
    entity_2 = split_arr[2]
    label = split_arr[3]
    lang = split_arr[-1]
    if len(split_arr) == 6:
        text = split_arr[4]
    elif len(split_arr) > 6:
        text = " ".join(split_arr[4:-1]).strip('\t')
    else:
        raise ValueError("No enough values to unpack")

    return label


def smiler_to_normalized_conll(input_path: str, output_path: str, input_lang: str):
    rel_to_line_dict = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = list(f)
        lines = lines[1:]  # header
        for i, line in tqdm(enumerate(lines)):
            rel = _get_rel_from_tsv_line(line)
            rel_to_line_dict[rel].append(line)

    # min_rel_count = min([len(k) for k in rel_to_line_dict])
    # min_rel_count = max(min_rel_count, 50)  # min 50 per rel
    # min_rel_count = min(min_rel_count, 200)  # max 200 per rel
    min_rel_count = 50

    normalized_lines_list = []
    for k, v in rel_to_line_dict.items():
        if len(v) >= min_rel_count:
            instances = v
            random.shuffle(v)
            normalized_lines_list.extend(instances[:min_rel_count])


    random.shuffle(normalized_lines_list)

    nlp = trankit.Pipeline(lang=input_lang, gpu=True, cache_dir='./cache')

    instance_list = []
    for i, line in tqdm(enumerate(normalized_lines_list)):
        instance = smiler_line_to_conll_dict_v2(line, input_lang, nlp)
        instance_list.append(instance)

    with open(output_path, 'x', encoding='utf-8') as f:
        json.dump(instance_list, f)


def create_normalized_test_datasets():
    # Because the actual test sets often contains less than 10 occurrences of each rel and even below that,
    # which cause the evaluation to be highly unstable, we create a new test set from the train set.
    # We only include relations that appear at least 50 times, to a maximum of 200 per rel, in order to keep the test
    # set small enough it is quick to evaluate it.
    input_files = [
        "experiments/datasets/relation_extraction/smiler/fa_corpora_train.tsv",
        "experiments/datasets/relation_extraction/smiler/ko_corpora_train.tsv",
        "experiments/datasets/relation_extraction/smiler/ar_corpora_train.tsv"
    ]

    for f in input_files:
        filename = os.path.basename(f)
        output_dir = f'experiments/processed_datasets/smiler/normalized_test_sets_50_per_rel/'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{filename}.json'
        output_path = output_path.replace("train", "test")
        if os.path.exists(output_path):
            print(f'File {output_path} already exists! Skipping!')
            continue

        if "ar" in filename:
            lang = "arabic"
        elif "de" in filename:
            lang = "german"
        elif "en" in filename:
            lang = "english"
        elif "es" in filename:
            lang = "spanish"
        elif "fa" in filename:
            lang = "persian"
        elif "fr" in filename:
            lang = "french"
        elif "it" in filename:
            lang = "italian"
        elif "ko" in filename:
            lang = "korean"
        elif "nl" in filename:
            lang = "dutch"
        elif "pl" in filename:
            lang = "polish"
        elif "pt" in filename:
            lang = "portuguese"
        elif "ru" in filename:
            lang = "russian"
        else:
            raise Exception("Unknown lang")
        smiler_to_normalized_conll(f, output_path, lang)


def create_small_datasets(input_path: str, size: int, input_lang: str, output_path: str):
    rel_to_line_dict = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = list(f)
        lines = lines[1:]  # header
        for i, line in tqdm(enumerate(lines)):
            rel = _get_rel_from_tsv_line(line)
            rel_to_line_dict[rel].append(line)
    rel_count = len(rel_to_line_dict.keys())
    wanted_rel_occurrence_count = int(size /rel_count)

    selected_lines_list = []
    left_lines_list = []
    for rel, instances in rel_to_line_dict.items():
        if len(instances) <= wanted_rel_occurrence_count:
            selected_lines_list.extend(instances)
        else:
            random.shuffle(instances)
            selected_instances = instances[:wanted_rel_occurrence_count]
            selected_lines_list.extend(selected_instances)
            left_lines_list.extend(instances[wanted_rel_occurrence_count:])

    missing_instances_count = size - len(selected_lines_list)
    random.shuffle(left_lines_list)
    selected_lines_list.extend(left_lines_list[:missing_instances_count])

    random.shuffle(selected_lines_list)

    nlp = trankit.Pipeline(lang=input_lang, gpu=True, cache_dir='./cache')

    instance_list = []
    for i, line in tqdm(enumerate(selected_lines_list)):
        instance = smiler_line_to_conll_dict_v2(line, input_lang, nlp)
        instance_list.append(instance)

    with open(output_path, 'x', encoding='utf-8') as f:
        json.dump(instance_list, f)

def create_small_train_dev_datasets(train_size: int, test_size: int):
    input_files = [
        "experiments/datasets/relation_extraction/smiler/en-full_corpora_train.tsv",
        "experiments/datasets/relation_extraction/smiler/en-full_corpora_test.tsv"
    ]

    output_dir = 'experiments/processed_datasets/smiler/train_normalized'

    os.makedirs(output_dir, exist_ok=True)

    for f in input_files:
        filename = os.path.basename(f)

        if "train" in filename:
            size = train_size
        else:
            size = test_size
        output_path = f'{output_dir}/{filename}-{size}.json'
        if os.path.exists(output_path):
            print(f'File {output_path} already exists! Skipping!')
            continue


        create_small_datasets(f, size, "english", output_path)

def filter_json_by_rel_set(rel_set: List, input_path: str, output_dir: str):
    filename = os.path.basename(input_path)
    filtered_instances = []
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for instance in data:
            assert len(instance['relations']) == 1
            rel = instance['relations'][0]['type']
            if rel in rel_set:
                filtered_instances.append(instance)

    with open(f'{output_dir}/{filename}', 'x', encoding='utf-8') as f:
        json.dump(filtered_instances, f)


def get_rel_set_from_json(path: str):
    rel_set = set()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for instance in data:
            for rel in instance['relations']:
                label = rel['type'].split("(")[0]
                rel_set.add(label)
    return rel_set


def create_normalized_test_datasets_with_only_32_rels():
    input_dir = "experiments/processed_datasets/smiler/test_normalized"
    input_files = glob.glob(f'{input_dir}/*')
    output_dir = "experiments/processed_datasets/smiler/test_normalized_32_rels"

    en_small_dataset = "experiments/processed_datasets/smiler/standard/en-small_corpora_train.tsv.json"
    rel_set = get_rel_set_from_json(en_small_dataset)

    for f in input_files:
        filter_json_by_rel_set(rel_set, f, output_dir)


if __name__ == '__main__':
    # create_normalized_test_datasets_with_only_32_rels()
    # create_small_train_dev_datasets(10000, 3600)
    # create_small_train_dev_datasets(35000, 3600)
    # create_standard_dataset()
    # create_reordered_dataset()
    # create_normalized_test_datasets()
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-s", "--size", required=True)

    args = argparser.parse_args()

    create_small_train_dev_datasets(args.size, 3600)
