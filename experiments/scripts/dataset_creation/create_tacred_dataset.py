import glob
import json
import os.path
import random
import string
from collections import defaultdict
from typing import Dict, List

import conllu
import trankit
from tqdm import tqdm

from reordering_package.ud_reorder_algo import UdReorderingAlgo


def tacred_instance_to_conll_dict(_id: str, tokenized_text: str, label: str) -> Dict:
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


def tacred_conllu_tree_to_conll_dict(tree: conllu.TokenList) -> Dict:
    _id = tree.metadata['id']
    label = tree.metadata['relation']
    tokens = json.loads(tree.metadata['token'])
    subj_start, subj_end = int(tree.metadata['subj_start']) - 1, int(tree.metadata['subj_end'])
    obj_start, obj_end = int(tree.metadata['obj_start']) - 1, int(tree.metadata['obj_end'])

    if subj_start < obj_start:
        ent_1_start, ent_1_end = subj_start, subj_end
        ent_2_start, ent_2_end = obj_start, obj_end
    else:
        ent_2_start, ent_2_end = subj_start, subj_end
        ent_1_start, ent_1_end = obj_start, obj_end

    tokens_arr = tokens
    prefix = " ".join(tokens_arr[:ent_1_start])
    entity_1 = " ".join(tokens_arr[ent_1_start:ent_1_end])
    midfix = " ".join(tokens_arr[ent_1_end:ent_2_start])
    entity_2 = " ".join(tokens_arr[ent_2_start:ent_2_end])
    postfix = " ".join(tokens_arr[ent_2_end:])

    sent = f'{prefix} <e1> {entity_1} </e1> {midfix} <e2> {entity_2} </e2> {postfix}'
    sent = sent.replace("  ", " ").strip()
    instance = tacred_instance_to_conll_dict(_id, sent, label)

    return instance

def reorder_relation_instance(
        instance: Dict,
        reorder_algo: UdReorderingAlgo,
        reorder_by_lang: str,
        parse_tree: conllu.TokenList) -> Dict:
    # TODO: In the future, generalize the Reordering Algo to an interface for the other reordering algorithm

    entities_list = []
    for ent in instance['entities']:
        ent_dict = {
            'surfaceform': "dummy",
            'inclusive_span': (ent['start'], ent['end'] - 1)
        }
        entities_list.append(ent_dict)

    sentence = " ".join(instance['tokens'])
    mapping = reorder_algo.get_entities_aware_reorder_mapping_with_parse_tree_input(sentence, reorder_by_lang,
                                                                                    entities_list,
                                                                                    parse_tree)

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



def reorder_tacred_file(input_path: str,
                      reorder_by_lang: str,
                      reorder_algo_type: UdReorderingAlgo.ReorderAlgo,
                      output_dir: str):
    reorder_algo = UdReorderingAlgo(reorder_algo_type, "english")
    reordered_instances = []
    instance_list = []
    error_dict = defaultdict(int)
    i=0
    with open(input_path, 'r', encoding='utf-8') as f:
        for tree in tqdm(conllu.parse_incr(f)):
            i += 1
            if i > 30000:
                break
            instance = tacred_conllu_tree_to_conll_dict(tree)
            instance_list.append(instance)
            try:
                reordered_instance = reorder_relation_instance(instance, reorder_algo, reorder_by_lang, tree)
                reordered_instances.append(reordered_instance)
            except Exception as e:
                raise e
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


def create_tacred_standard_dataset():
    input_files = [
        "experiments/datasets/relation_extraction/tacred_annotated/en/train.conllu",
        "experiments/datasets/relation_extraction/tacred_annotated/en/dev.conllu",
    ]

    output_dir = f'experiments/processed_datasets/tacred_small/standard/'
    os.makedirs(output_dir, exist_ok=True)

    for f_path in input_files:
        filename = os.path.basename(f_path).split(".")[0]
        instances = []
        i=0
        with open(f_path, 'r', encoding='utf-8') as f:
            for tree in tqdm(conllu.parse_incr(f)):
                i+=1
                if i > 30000:
                    break
                instance = tacred_conllu_tree_to_conll_dict(tree)
                instances.append(instance)

        output_path = f'{output_dir}/{filename}.conllu.json'
        with open(output_path, 'x', encoding='utf-8') as f:
            json.dump(instances, f)


def create_tacred_reordered_dataset():
    file_paths = [
        "experiments/datasets/relation_extraction/tacred_annotated/en/train.conllu",
        "experiments/datasets/relation_extraction/tacred_annotated/en/dev.conllu",
    ]

    for file_path in file_paths:
        for lang in ["korean", "persian", "turkish"]:
            output_dir = f'experiments/processed_datasets/tacred_small/english_reordered_by_{lang}/'
            os.makedirs(output_dir, exist_ok=True)

            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.RASOOLINI]: #UdReorderingAlgo.ReorderAlgo.HUJI
                print(
                    f'Creating seq2seq dataset. lang: {lang}, file: {os.path.basename(file_path)}, algorithm: {reorder_algo.name}')
                reorder_tacred_file(file_path,
                                  lang,
                                  reorder_algo,
                                  output_dir)


if __name__ == '__main__':
    # create_tacred_standard_dataset()
    create_tacred_reordered_dataset()
