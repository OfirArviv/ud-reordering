import os.path
from typing import List

import conllu


# Code adapted from: https://github.com/bcmi220/seq2seq_parser/blob/master/data/scripts/make_dp_dataset.py\

def conllu_to_seq2seq(ud_file_path: str, output_file_path: str):
    with open(ud_file_path, 'r', encoding='utf-8') as f, \
            open(output_file_path, 'x', encoding='utf-8') as out_f:
        for sent in conllu.parse_incr(f):
            sent_input_seq = " ".join([node['form'] for node in sent])
            sent_target_seq_arr = []
            for node in sent:
                if not isinstance(node['id'], int):
                    continue
                dep_ind = int(node['id'])
                head_ind = int(node['head'])
                if dep_ind > head_ind:
                    tag = 'L' + str(abs(dep_ind - head_ind))
                else:
                    tag = 'R' + str(abs(dep_ind - head_ind))
                dep_id = node['deprel']
                dep_id = dep_id.split(":")[0]
                tag = tag + " " + dep_id
                sent_target_seq_arr.append(tag)
            sent_target_seq = " ".join(sent_target_seq_arr)
            out_f.write(f'{sent_input_seq}\t{sent_target_seq}\n')


def create_vanilla_seq2seq_dataset_script():
    dataset_root_dir = "experiments/datasets/ud/ud-treebanks-v2.10"
    selected_datasets_list = [
        f'{dataset_root_dir}/UD_English-EWT/en_ewt-ud-train.conllu',
        f'{dataset_root_dir}/UD_English-EWT/en_ewt-ud-dev.conllu',
        f'{dataset_root_dir}/UD_Hindi-PUD/hi_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_Indonesian-PUD/id_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_Japanese-PUD/ja_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_Korean-PUD/ko_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_Persian-Seraji/fa_seraji-ud-test.conllu',
        f'{dataset_root_dir}/UD_Spanish-PUD/es_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_Thai-PUD/th_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_Turkish-PUD/tr_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_French-PUD/fr_pud-ud-test.conllu',
        f'{dataset_root_dir}/UD_German-PUD/de_pud-ud-test.conllu'
    ]
    output_dir = 'experiments/processed_datasets/ud/seq2seq_standard'
    os.makedirs(output_dir, exist_ok=True)

    for d in selected_datasets_list:
        basename = os.path.basename(d).split(".")[0]
        conllu_to_seq2seq(d, f'{output_dir}/{basename}.tsv')


def create_vocab_from_seq2seq_file(file_path_list: List[str], output_dir: str) -> None:
    ontology_tokens = ['@@UNKNOWN@@', '@@PADDING@@', '@start@', '@end@']
    ud_labels = set()
    pos_toks = set()

    for file_path in file_path_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                temp = line.split('\t')
                if len(temp) != 2:
                    continue
                source_seq, target_seq = temp
                target_seq = target_seq.strip('\n')
                target_tokens: List[str] = target_seq.split()
                for token in target_tokens:
                    if token.startswith('R') or token.startswith('L'):
                        ud_labels.add(token)
                    else:
                        pos_toks.add(token)

    with open(f'{output_dir}/target_tokens.txt', 'x', encoding='utf-8') as f:
        for token in ontology_tokens:
            f.write(f'{token}\n')

        for token in sorted(list(ud_labels)):
            f.write(f'{token}\n')

        for token in sorted(list(pos_toks)):
            f.write(f'{token}\n')

    with open(f'{output_dir}/non_padded_namespaces.txt', 'x', encoding='utf-8') as f:
        for token in ["*tags", "*labels"]:
            f.write(f'{token}\n')


# def create_reordered_datasets_script():
#     for lang in ['hindi', 'thai', 'french', 'spanish', 'german']:
#         output_dir = f'experiments/processed_datasets/mtop/'
#         if use_pointers:
#             output_dir += "pointers_format/"
#         output_dir += f'english_reordered_by_{lang}'
#         os.makedirs(output_dir, exist_ok=True)
#         for split in ['train', 'eval', 'test']:
#             for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
#                 print(f'Creating seq2seq dataset. lang: {lang}, split: {split}, algorithm: {reorder_algo.name}')
#                 create_mtop_seq2seq_dataset(f'experiments/datasets/top/mtop/en/{split}.txt',
#                                             "english",
#                                             use_pointers,
#                                             use_decoupled_format,
#                                             output_dir,
#                                             reorder_algo,
#                                             lang)


if __name__ == "__main__":
    # create_vanilla_seq2seq_dataset_script()

    # Create vocab
    create_vocab_from_seq2seq_file([
        "experiments/processed_datasets/ud/seq2seq_standard/en_ewt-ud-train.tsv",
        "experiments/processed_datasets/ud/seq2seq_standard/en_ewt-ud-dev.tsv",
    ],
        "experiments/vocabs/ud/"
    )

     #  TODO: Decide on zero-shot it and then on the datasets to use

##### UNFINISHED !!!!!!!