import glob
import json
import os.path
import random
from collections import defaultdict
from typing import Dict

import trankit
from tqdm import tqdm

from reordering_package.ud_reorder_algo import UdReorderingAlgo


def smiler_line_to_conll_dict(line: str, input_lang: str, nlp) -> Dict:
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

    text = text.strip()

    tokens = []
    entity_1_start_idx = None
    entity_1_end_idx = None
    entity_2_start_idx = None
    entity_2_end_idx = None
    i = 0
    for tok in text.split():
        entity_1_end_flag = False
        entity_2_end_flag = False
        if '<e1>' in tok:
            assert entity_1_start_idx is None
            entity_1_start_idx = i
            tok = " ".join(tok.split("<e1>")).strip()
        if '<e2>' in tok:
            assert entity_2_start_idx is None
            entity_2_start_idx = i
            tok = "".join(tok.split("<e2>")).strip()
        if '</e1>' in tok:
            assert entity_1_start_idx is not None
            assert entity_1_end_idx is None
            tok = "".join(tok.split("</e1>")).strip()
            entity_1_end_flag = True
        if '</e2>' in tok:
            assert entity_2_start_idx is not None
            assert entity_2_end_idx is None
            tok = "".join(tok.split("</e2>")).strip()
            entity_2_end_flag = True

        doc = nlp.tokenize(tok, is_sent=True)
        doc_tok_list = [t['text'] for t in doc['tokens']]
        i += len(doc_tok_list)
        tokens.extend(doc_tok_list)

        if entity_1_end_flag:
            entity_1_end_idx = i - 1
        if entity_2_end_flag:
            entity_2_end_idx = i - 1

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

    relations = [
        {
            "type": label,
            "head": 0,
            "tail": 1
        }
    ]

    instance = {
        "orig_id": _id,
        "tokens": tokens,
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

        if len(lines) > 10000:
            random.shuffle(lines)
            lines = lines[:10000]
        for i, line in tqdm(enumerate(lines)):
            instance = smiler_line_to_conll_dict(line, input_lang, nlp)
            instance_list.append(instance)

    with open(output_path, 'x', encoding='utf-8') as f:
        json.dump(instance_list, f)


def smiler_to_rel_type(input_path: str, output_path: str):
    rel_set = set()
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header

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

            rel_set.add(label)

    rel_dict = {}
    for rel in rel_set:
        rel_dict[rel] = {
            "short": rel,
            "verbose": rel,
            "symmetric": rel in ["no_relation"]
        }

    entities_dict = {
        "dummy": {
            "short": "dummy",
            "verbose": "dummy",
        }
    }

    types_dict = {
        "entities": entities_dict,
        "relations": rel_dict
    }

    with open(output_path, 'x', encoding='utf-8') as f:
        json.dump(types_dict, f)


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
            print(str(e))
            error_dict[str(e)] += 1
            reordered_instances.append(instance)

    print(error_dict)

    filename = os.path.basename(input_path).split(".")[0]
    output_file_path = f'{output_dir}/{filename}'

    output_file_path += f'_reordered_by_{reorder_by_lang}_{reorder_algo_type.name}'
    with open(f'{output_file_path}.tsv', 'x', encoding='utf-8') as f:
        json.dump(reordered_instances, f)

    output_file_path += f'_combined'
    with open(f'{output_file_path}.tsv', 'x', encoding='utf-8') as f:
        temp = instance_list + reordered_instances
        random.shuffle(temp)
        json.dump(temp, f)


def create_standard_dataset():
    main_dir = "re_tryout/smiler_dataset"
    input_files = glob.glob(f'{main_dir}/*.tsv')
    input_files = [
        # "re_tryout/smiler_dataset/en_corpora_train.tsv",
        # "re_tryout/smiler_dataset/en_corpora_test.tsv",
        "re_tryout/smiler_dataset/ko_corpora_test.tsv",
        "re_tryout/smiler_dataset/fa_corpora_test.tsv",
        "re_tryout/smiler_dataset/ar_corpora_test.tsv"
    ]
    for f in input_files:
        filename = os.path.basename(f)
        output_path = f're_tryout/simlier_dataset_conll_format/standard/{filename}.json'

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


def create_type_files():
    en_train_path = "re_tryout/smiler_dataset/en-full_corpora_train.tsv"
    output_path = "re_tryout/simlier_dataset_conll_format/types/types.json"
    smiler_to_rel_type(en_train_path, output_path)


def create_reordered_dataset():
    file_paths = [
        "re_tryout/simlier_dataset_conll_format/standard/en_corpora_train.tsv.json",
        "re_tryout/simlier_dataset_conll_format/standard/en_corpora_test.tsv.json"
    ]

    for file_path in file_paths:
        for lang in ["korean", "arabic", "persian"]:
            output_dir = f're_tryout/simlier_dataset_conll_format/english_reordered_by_{lang}/'
            os.makedirs(output_dir, exist_ok=True)

            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(
                    f'Creating seq2seq dataset. lang: {lang}, file: {os.path.basename(file_path)}, algorithm: {reorder_algo.name}')
                reorder_file(file_path,
                             lang,
                             reorder_algo,
                             output_dir)


# create_type_files()


create_standard_dataset()
create_reordered_dataset()
