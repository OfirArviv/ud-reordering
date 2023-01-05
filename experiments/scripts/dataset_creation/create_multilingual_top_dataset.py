import os
from collections import defaultdict
import random
from typing import Optional
from tqdm import tqdm

from experiments.scripts.dataset_creation.create_mtop_dataset import _get_parse_tree_from_str, \
    parse_tree_to_pointer_format, get_parse_tree_str_from_pointer_tree, reorder_mtop_pointer_tree, \
    create_vocab_from_seq2seq_file
from reordering_package.ud_reorder_algo import UdReorderingAlgo


def multilingual_top_format_to_mtop_format(target_mrl: str, nl_line: str) -> str:
    # The only thing this function does is to replace the closing parentheses of format '](node type)' with plain ']'.
    nl_arr = nl_line.split()
    new_str_arr = []
    for token in target_mrl.split():
        if token.startswith("]"):
            new_str_arr.append("]")
        elif token.startswith("@ptr"):
            tok_idx = int(token.split("@ptr")[1])
            new_str_arr.append(nl_arr[tok_idx])
        else:
            new_str_arr.append(token)
    return " ".join(new_str_arr)


def create_multilingual_top_seq2seq_dataset(input_mrl_file_path: str,
                                            input_nl_file_path: str,
                                            input_lang: str,
                                            use_pointer_format: bool,
                                            use_decoupled_format: bool,
                                            output_dir: str,
                                            reorder_algo: Optional[UdReorderingAlgo.ReorderAlgo] = None,
                                            reorder_by_lang: Optional[str] = None):
    if reorder_algo:
        ud_reorder_algo = UdReorderingAlgo(reorder_algo, input_lang)

    seq2seq_std_strings = []
    seq2seq_reordered_strings = []

    error_dict = defaultdict(int)
    with open(input_mrl_file_path, "r", encoding='utf-8') as mrl_file, \
            open(input_nl_file_path, "r", encoding='utf-8') as nl_file:
        for mrl_line, nl_line in tqdm(zip(mrl_file, nl_file)):
            mrl_line = mrl_line.strip("\n")
            nl_line = nl_line.strip("\n")
            if not mrl_line or not nl_line:
                continue

            source_sequence = nl_line
            decoupled_form_string_pointer_format = mrl_line

            target_sequence = multilingual_top_format_to_mtop_format(decoupled_form_string_pointer_format, nl_line)
            try:
                parse_tree = _get_parse_tree_from_str(target_sequence)
                pointers_parse_tree = parse_tree_to_pointer_format(parse_tree, source_sequence)
            except Exception as e:
                error_dict[str(e)] += 1
                continue

            if not use_decoupled_format:
                raise NotImplemented()
                # TODO: Add function that uses pointers format
                # parse_tree = _partial_format_tree_to_full_format_tree(parse_tree, source_sequence)

            parse_tree_str = get_parse_tree_str_from_pointer_tree(pointers_parse_tree,
                                                                  source_sequence,
                                                                  use_pointer_format)

            if reorder_by_lang:
                try:
                    reordered_source_sequence, reordered_pointer_parse_tree = \
                        reorder_mtop_pointer_tree(
                            pointers_parse_tree,
                            ud_reorder_algo,
                            source_sequence,
                            reorder_by_lang,
                            use_decoupled_format)

                    reordered_parse_tree_str = get_parse_tree_str_from_pointer_tree(reordered_pointer_parse_tree,
                                                                                    reordered_source_sequence,
                                                                                    use_pointer_format)
                    seq2seq_reordered_strings.append(f'{reordered_source_sequence}\t{reordered_parse_tree_str}\n')
                except Exception as e:
                    error_dict[f'error reordering tree - {str(e)}'] += 1
                    print(e)
                    seq2seq_reordered_strings.append(f'{source_sequence}\t{parse_tree_str}\n')

            seq2seq_std_strings.append(f'{source_sequence}\t{parse_tree_str}\n')

    print(error_dict)

    filename = os.path.basename(input_mrl_file_path).split(".")[0]

    output_file_path = f'{output_dir}/{input_lang}_{filename}'
    output_file_path += "_decoupled_format" if use_decoupled_format else "full_format"

    if reorder_by_lang:
        output_file_path += f'_reordered_by_{reorder_by_lang}_{reorder_algo.name}'
        with open(f'{output_file_path}.tsv', 'x', encoding='utf-8') as f:
            for s in seq2seq_reordered_strings:
                f.write(f'{s}')

        output_file_path += f'_combined'
        combined_seq2seq_strings = seq2seq_std_strings + seq2seq_reordered_strings
        random.shuffle(combined_seq2seq_strings)
        with open(f'{output_file_path}.tsv', 'x', encoding='utf-8') as f:
            for s in combined_seq2seq_strings:
                f.write(f'{s}')
    else:
        with open(f'{output_file_path}.tsv', 'x', encoding='utf-8') as f:
            for s in seq2seq_std_strings:
                f.write(f'{s}')


def create_english_dataset_script_multilingual_top(use_pointers: bool):
    use_decoupled_format = True
    output_dir = "experiments/processed_datasets/multilingual_top/"
    if use_pointers:
        output_dir += "pointers_format/"
    output_dir += "/standard/"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = "experiments/datasets/top/multilingual-top-main/processed_data/en"
    for split in ['train', 'dev', 'test']:
        print(f'Creating seq2seq dataset. lang: en, split: {split}')
        create_multilingual_top_seq2seq_dataset(f'{dataset_dir}/{split}.mrl',
                                                f'{dataset_dir}/{split}.nl',
                                                "english",
                                                use_pointers,
                                                use_decoupled_format,
                                                output_dir,
                                                None,
                                                None)


def create_test_datasets_script_multilingual_top(use_pointers: bool):
    use_decoupled_format = True
    output_dir = "experiments/processed_datasets/multilingual_top/"
    if use_pointers:
        output_dir += "pointers_format/"
    output_dir += "/standard/"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = "experiments/datasets/top/multilingual-top-main/processed_data/"
    for lang in ['it', 'ja']:
        print(f'Creating seq2seq dataset. lang: {lang}, split: test')
        create_multilingual_top_seq2seq_dataset(
            f'{dataset_dir}/{lang}/test.mrl',
            f'{dataset_dir}/{lang}/test.nl',
            lang,
            use_pointers,
            use_decoupled_format,
            output_dir,
            None,
            None)


def create_reordered_datasets_scripts_multilingual_top(use_pointers: bool):
    use_decoupled_format = True
    for lang in ['japanese', 'italian']:
        output_dir = "experiments/processed_datasets/multilingual_top/"
        if use_pointers:
            output_dir += "pointers_format/"
        output_dir += f'english_reordered_by_{lang}'
        os.makedirs(output_dir, exist_ok=True)
        dataset_dir = "experiments/datasets/top/multilingual-top-main/processed_data/en"
        for split in ['train', 'dev', 'test']:
            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(f'Creating seq2seq dataset. lang: {lang}, split: {split}, algorithm: {reorder_algo.name}')
                create_multilingual_top_seq2seq_dataset(f'{dataset_dir}/{split}.mrl',
                                                        f'{dataset_dir}/{split}.nl',
                                                        "english",
                                                        use_pointers,
                                                        use_decoupled_format,
                                                        output_dir,
                                                        reorder_algo,
                                                        lang)


def create_rasoolini_huji_combined_datasets():
    pointer_dataset_path = 'experiments/processed_datasets/multilingual_top/pointers_format/'
    for lang in ['japanese', 'italian']:
        for split in ['dev', 'train', 'test']:
            standard_path = f'{pointer_dataset_path}/standard/english_{split}_decoupled_format.tsv'
            reordered_by_lang_dir = f'{pointer_dataset_path}/english_reordered_by_{lang}/'
            reordered_huji = f'{reordered_by_lang_dir}/english_{split}_decoupled_format_reordered_by_{lang}_HUJI.tsv'
            reordered_rasoolini = f'{reordered_by_lang_dir}/english_{split}_decoupled_format_reordered_by_{lang}_RASOOLINI.tsv'

            with open(standard_path, 'r', encoding='utf-8') as std_f,\
                open(reordered_huji, 'r', encoding='utf-8') as huji_f,\
                    open(reordered_rasoolini, 'r', encoding='utf-8') as rasoolini_f:
                # std_list = list(std_f)
                huji_list = list(huji_f)
                rasoolini_list = list(rasoolini_f)

            output_path = f'{reordered_by_lang_dir}/english_{split}_decoupled_format_reordered_by_{lang}_HUJI_RASOOLINI.tsv'

            output_list = huji_list + rasoolini_list
            random.shuffle(output_list)

            with open(output_path, 'x', encoding='utf-8') as o_f:
                for l in output_list:
                    o_f.write(l)

if __name__ == "__main__":
    # Create English standard dataset
    create_english_dataset_script_multilingual_top(True)

    # Create Reordered Dataset
    create_reordered_datasets_scripts_multilingual_top(True)

    create_rasoolini_huji_combined_datasets()

    # Create Test Dataset
    create_test_datasets_script_multilingual_top(True)

    # Create vocab
    create_vocab_from_seq2seq_file([
        "experiments/processed_datasets/multilingual_top/pointers_format/standard/english_train_decoupled_format.tsv",
        "experiments/processed_datasets/multilingual_top/pointers_format/standard/english_dev_decoupled_format.tsv",
    ],
        "experiments/vocabs/multilingual_top_pointers/"
    )
