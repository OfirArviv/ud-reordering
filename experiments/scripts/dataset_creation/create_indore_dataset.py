import glob
import json
import os.path
import random
import string
from collections import defaultdict
from typing import Dict, List

import trankit
from tqdm import tqdm

from reordering_package.ud_reorder_algo import UdReorderingAlgo


def instance_to_conll_dict(_id: str, text: str, label: str, nlp) -> Dict:
    assert "<e1>" and "</e1>" in text
    assert "<e2>" and "</e2>" in text

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

    # if any([s in entity_1_str for s in string.punctuation]):
    #     print("\n" + " ".join(entity_1_str))
    #
    # if any([s in entity_2_str for s in string.punctuation]):
    #     print("\n" + " ".join(entity_2_str))

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


def reorder_json_file(input_path: str,
                      reorder_by_lang: str,
                      reorder_algo_type: UdReorderingAlgo.ReorderAlgo,
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



def rlex_to_conll(input_path: str, output_path: str, input_lang: str):
    nlp = trankit.Pipeline(lang=input_lang, gpu=True, cache_dir='./cache')

    instance_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = list(f)
        lines = [l.strip('\n') for l in lines]
        lines = [l for l in lines if len(l) > 0]
        lines = [l.strip().strip() for l in lines if len(l) > 0]
        lines_indexes = [i for i in range(len(lines)) if i % 2 == 0]
        fixed_lines = [f'{lines[i]}\t{lines[i + 1]}' for i in lines_indexes]
        random.shuffle(fixed_lines)

        for i, line in tqdm(enumerate(fixed_lines)):
            _id, text, rel = line.split('\t')
            rel = rel.split("(")[0]
            text = text.strip('"')
            try:
                instance = instance_to_conll_dict(_id, text, rel, nlp)
            except:
                print(f'error with text: {text}')
                continue
            instance_list.append(instance)

    with open(output_path, 'x', encoding='utf-8') as f:
        json.dump(instance_list, f)

def create_rlex_standard_dataset():
    input_files = [
        # "experiments/datasets/relation_extraction/rlex/RELX_en.txt",
        "experiments/datasets/relation_extraction/rlex/RELX_tr.txt"
    ]

    output_dir = f'experiments/processed_datasets/rlex/standard/'
    os.makedirs(output_dir, exist_ok=True)

    for f in input_files:
        filename = os.path.basename(f).split(".")[0]

        if "en" in filename:
            lang = "english"
        elif "tr" in filename:
            lang = "turkish"
        else:
            raise NotImplementedError()

        instances = indore_to_conll_instance_list(f, lang)
        random.shuffle(instances)

        if lang == "english":
            train_split = instances[:(0.9 * len(instances))]
            dev_split = instances[(0.9 * len(instances)):]

            train_output_path = f'{output_dir}/{filename}_train.txt.json'
            dev_output_path = f'{output_dir}/{filename}_train.txt.json'

            with open(train_output_path, 'x', encoding='utf-8') as f:
                json.dump(train_split, f)
            with open(dev_output_path, 'x', encoding='utf-8') as f:
                json.dump(dev_split, f)
        else:
            output_path = f'{output_dir}/{filename}.txt.json'
            with open(output_path, 'x', encoding='utf-8') as f:
                json.dump(instances, f)


def indore_to_conll_instance_list(input_path: str, input_lang: str) -> List[Dict]:
    nlp = trankit.Pipeline(lang=input_lang, gpu=True, cache_dir='./cache')

    instance_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = list(f)
        lines = [l.strip('\n') for l in lines]
        lines = [l.strip().strip() for l in lines if len(l) > 0]
        random.shuffle(lines)

        for i, line in tqdm(enumerate(lines)):
            rel, text, ent1, ent2 = line.split('\t')
            text = text.strip('"')

            instance = instance_to_conll_dict(i, text, rel, nlp)
            instance_list.append(instance)

    return instance_list


def create_indore_standard_dataset():
    input_files = [
        "experiments/datasets/relation_extraction/IndoRE/english_indore.tsv",
        "experiments/datasets/relation_extraction/IndoRE/hindi_indore.tsv"
    ]

    output_dir = f'experiments/processed_datasets/indore/standard/'
    os.makedirs(output_dir, exist_ok=True)

    for f in input_files:
        filename = os.path.basename(f).split(".")[0]

        if "english" in filename:
            lang = "english"
        elif "hindi" in filename:
            lang = "hindi"
        else:
            raise NotImplementedError()

        instances = indore_to_conll_instance_list(f, lang)
        random.shuffle(instances)

        if lang == "english":
            train_split = instances[:int(0.9*len(instances))]
            dev_split = instances[int(0.9*len(instances)):]

            train_output_path = f'{output_dir}/{filename}_train.tsv.json'
            dev_output_path = f'{output_dir}/{filename}_dev.tsv.json'

            with open(train_output_path, 'x', encoding='utf-8') as f:
                json.dump(train_split, f)
            with open(dev_output_path, 'x', encoding='utf-8') as f:
                json.dump(dev_split, f)
        else:
            output_path = f'{output_dir}/{filename}.tsv.json'
            with open(output_path, 'x', encoding='utf-8') as f:
                json.dump(instances, f)


def create_indore_reordered_dataset():
    file_paths = [
        "experiments/processed_datasets/indore/standard/english_indore_train.tsv.json",
        "experiments/processed_datasets/indore/standard/english_indore_dev.tsv.json"
    ]

    for file_path in file_paths:
        for lang in ["hindi"]:
            output_dir = f'experiments/processed_datasets/indore/english_reordered_by_{lang}/'
            os.makedirs(output_dir, exist_ok=True)

            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(
                    f'Creating seq2seq dataset. lang: {lang}, file: {os.path.basename(file_path)}, algorithm: {reorder_algo.name}')
                reorder_json_file(file_path,
                             lang,
                             reorder_algo,
                             output_dir)


def create_klue_re_dataset():
    input_files = [
        "experiments/datasets/relation_extraction/klue_re/klue-re-v1.1/klue-re-v1.1_dev.json"
    ]

    output_dir = f'experiments/processed_datasets/klue_re/standard/'
    os.makedirs(output_dir, exist_ok=True)

    for f in input_files:
        filename = os.path.basename(f).split(".")[0]
        conll_instances_list = []
        nlp = trankit.Pipeline(lang="korean", gpu=True, cache_dir='./cache')
        with open(f, 'r', encoding='utf-8') as i_f:
            data = json.load(i_f)

            random.shuffle(data)
            data = data[:1000]

            for instance in tqdm(data):
                _id = instance['guid']
                label = instance['label']
                sentence = instance['sentence']

                subj_word = instance['subject_entity']['word']
                subj_start, subj_end = instance['subject_entity']['start_idx'], instance['subject_entity']['end_idx'] + 1
                assert subj_word == sentence[subj_start:subj_end]

                obj_word = instance['object_entity']['word']
                obj_start, obj_end = instance['object_entity']['start_idx'], instance['object_entity']['end_idx'] + 1
                assert obj_word == sentence[obj_start:obj_end]

                if subj_start < obj_start:
                    ent_1_start, ent_1_end = subj_start, subj_end
                    ent_2_start, ent_2_end = obj_start, obj_end
                else:
                    ent_2_start, ent_2_end = subj_start, subj_end
                    ent_1_start, ent_1_end = obj_start, obj_end

                prefix = sentence[:ent_1_start]
                entity_1 = sentence[ent_1_start:ent_1_end]
                midfix = sentence[ent_1_end:ent_2_start]
                entity_2 = sentence[ent_2_start:ent_2_end]
                postfix = sentence[ent_2_end:]

                sent = f'{prefix} <e1> {entity_1} </e1> {midfix} <e2> {entity_2} </e2> {postfix}'
                sent = sent.replace("  ", " ").strip()

                conll_dict = instance_to_conll_dict(_id, sent, label, nlp)
                conll_instances_list.append(conll_dict)

        output_path = f'{output_dir}/{filename}.json'
        with open(output_path, 'x', encoding='utf-8') as f:
            json.dump(conll_instances_list, f)

if __name__ == '__main__':
    create_klue_re_dataset()
    # create_indore_standard_dataset()
    # create_indore_reordered_dataset()

    # create_rlex_standard_dataset()
    # create_reordered_dataset()
    # create_normalized_test_datasets()
