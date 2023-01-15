import glob
import os.path
from typing import List

import conllu


# Code adapted from: https://github.com/bcmi220/seq2seq_parser/blob/master/data/scripts/make_dp_dataset.py
def conllu_to_seq2seq(ud_file_path: str, output_file_path: str, use_pointer_format: bool):
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
                if use_pointer_format:
                    tag = f'@ptr{head_ind}'
                else:
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


def create_seq2seq_dataset_script(use_pointer_format: bool):
    conllu_dataset_root_dir = "experiments/processed_datasets/ud/conllu_format/"
    seq2seq_dataset_root_dir = f'experiments/processed_datasets/ud/seq2seq{"_pointer_format" if use_pointer_format else ""}/'
    os.makedirs(seq2seq_dataset_root_dir, exist_ok=True)

    _dir, subdirs, files = list(os.walk(conllu_dataset_root_dir))[0]
    for subdir in subdirs:
        subdir_path = os.path.join(_dir, subdir)
        for f_path in glob.glob(f'{subdir_path}/*conllu'):
            basename = os.path.basename(f_path)
            output_subdir = os.path.join(seq2seq_dataset_root_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f'{basename}.tsv')
            if os.path.exists(output_path):
                print(f'{output_path} already exists! Skipping!')
                continue
            conllu_to_seq2seq(f_path, output_path, use_pointer_format)


def create_vocab_from_seq2seq_file(file_path_list: List[str], output_dir: str) -> None:
    ontology_tokens = ['@@UNKNOWN@@', '@@PADDING@@', '@start@', '@end@']
    pointers_tokens = set()

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
                    if token.startswith('@ptr'):
                        pointers_tokens.add(token)
                    else:
                        if token not in ontology_tokens:
                            ontology_tokens.append(token)


    with open(f'{output_dir}/target_tokens.txt', 'w', encoding='utf-8') as f:
        for token in ontology_tokens:
            f.write(f'{token}\n')

        pointer_tokens_size = 150
        assert pointer_tokens_size > len(pointers_tokens)
        for i in range(pointer_tokens_size):
            f.write(f'@ptr{i}\n')

    with open(f'{output_dir}/non_padded_namespaces.txt', 'w', encoding='utf-8') as f:
        for token in ["*tags", "*labels"]:
            f.write(f'{token}\n')
if __name__ == "__main__":
    # create_seq2seq_dataset_script(True)

    # Create vocab
    create_vocab_from_seq2seq_file([
        "experiments/processed_datasets/ud/seq2seq_pointer_format/standard/en_ewt-ud-train.conllu.tsv",
        "experiments/processed_datasets/ud/seq2seq_pointer_format/standard/en_ewt-ud-dev.conllu.tsv",
    ],
        "experiments/vocabs/ud_pointers/"
    )