import glob
import os.path
import shutil
from collections import defaultdict
import random
import conllu
from tqdm import tqdm

from reordering_package.huji_ud_reordering import UDLib
from reordering_package.ud_reorder_algo import UdReorderingAlgo


def conllu_to_UDLib_format(input_tree: conllu.TokenList) -> UDLib.UDTree:
    input_tree_str = input_tree.serialize()
    assert input_tree_str[-1] == "\n"
    input_tree_str = input_tree_str[:-1]
    input_tree: UDLib.UDTree = UDLib.UDTree(*UDLib.conll2graph(input_tree_str))

    return input_tree


def create_reorder_ud_dataset(input_file_path: str,
                              input_lang: str,
                              output_dir: str,
                              reorder_algo: UdReorderingAlgo.ReorderAlgo,
                              reorder_by_lang: str):
    filename = os.path.basename(input_file_path).split(".")[0]
    output_file_path = f'{output_dir}/{input_lang}_{filename}'
    output_file_path += f'_reordered_by_{reorder_by_lang}_{reorder_algo.name}'
    output_file_path_combined = output_file_path + f'_combined'
    if os.path.exists(f'{output_file_path}.conllu') and os.path.exists(f'{output_file_path_combined}.conllu'):
        print(f'file already exists')
        return

    ud_reorder_algo = UdReorderingAlgo(reorder_algo, input_lang)

    std_trees = []
    reordered_trees = []

    error_dict = defaultdict(int)
    with open(input_file_path, "r", encoding='utf-8') as input_file:
        for tokenlist in tqdm(conllu.parse_incr(input_file)):
            new_tok_list = []
            for tok in tokenlist:
                if not isinstance(tok["id"], int):
                    assert isinstance(tok["id"], tuple)
                    continue
                if ":" in tok['deprel']:
                    tok['deprel'] = tok['deprel'].split(":")[0]
                new_tok_list.append(tok)
            conllu_token_list = conllu.TokenList(new_tok_list, {})
            conllu_token_list.metadata = tokenlist.metadata

            try:
                if reorder_algo == UdReorderingAlgo.ReorderAlgo.HUJI or reorder_algo == UdReorderingAlgo.ReorderAlgo.HUJI_GUROBI:
                    _, reordered_tree = ud_reorder_algo._get_huji_reorder_mapping(conllu_token_list, reorder_by_lang)
                elif reorder_algo == UdReorderingAlgo.ReorderAlgo.RASOOLINI:
                    _, reordered_tree = ud_reorder_algo._get_rasoolini_reorder_mapping(conllu_token_list,
                                                                                       reorder_by_lang)
                    reordered_tree = conllu_to_UDLib_format(reordered_tree)
                else:
                    raise NotImplementedError()

                if reordered_tree is None:
                    raise Exception("Cannot reorder sentence")

                reordered_trees.append(reordered_tree)
            except Exception as e:
                error_dict[f'error reordering tree - {str(e)}'] += 1
                tree = conllu_to_UDLib_format(conllu_token_list)
                reordered_trees.append(tree)

            tree = conllu_to_UDLib_format(conllu_token_list)
            std_trees.append(tree)

    print(error_dict)

    filename = os.path.basename(input_file_path).split(".")[0]

    output_file_path = f'{output_dir}/{input_lang}_{filename}'

    output_file_path += f'_reordered_by_{reorder_by_lang}_{reorder_algo.name}'
    with open(f'{output_file_path}.conllu', 'x', encoding='utf-8') as f:
        print('\n\n'.join(str(t) for t in reordered_trees), file=f)

    output_file_path += f'_combined'
    combined_trees = reordered_trees + std_trees
    random.shuffle(combined_trees)
    with open(f'{output_file_path}.conllu', 'x', encoding='utf-8') as f:
        print('\n\n'.join(str(t) for t in combined_trees), file=f)


def create_reordered_datasets_script():
    for lang in ['hindi', 'thai', 'french', 'spanish', 'german', 'persian', 'korean']:
        output_dir = f'experiments/processed_datasets/ud/conllu_format/'
        output_dir += f'english_reordered_by_{lang}'
        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'dev']:
            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(f'Creating seq2seq dataset. lang: {lang}, split: {split}, algorithm: {reorder_algo.name}')
                create_reorder_ud_dataset(
                    f'experiments/processed_datasets/ud/conllu_format/standard/en_ewt-ud-{split}.conllu',
                    "english",
                    output_dir,
                    reorder_algo,
                    lang)


def copy_standard_datasets():
    english_datasets = [
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-train.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-dev.conllu"
    ]

    selected_test_datasets = [
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_French-PUD/fr_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_German-PUD/de_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Spanish-PUD/es_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Persian-Seraji/fa_seraji-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Hindi-PUD/hi_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Thai-PUD/th_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Turkish-PUD/tr_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Indonesian-PUD/id_pud-ud-test.conllu",
        "experiments/datasets/ud/ud-treebanks-v2.10/UD_Korean-PUD/ko_pud-ud-test.conllu"
    ]

    output_dir = "experiments/processed_datasets/ud/conllu_format/standard/"
    os.makedirs(output_dir, exist_ok=True)

    for f_path in english_datasets + selected_test_datasets:
        output_path = f'{output_dir}/{os.path.basename(f_path)}'

        if not os.path.exists(output_path):
            print(f'copying {f_path}')
            shutil.copyfile(f_path, output_path)


def create_ud_vocab():
    conllu_dataset_root_dir = "experiments/processed_datasets/ud/conllu_format/standard/"
    dep_id_set = set()
    for ud_file_path in glob.glob(f'{conllu_dataset_root_dir}/*.conllu'):
        with open(ud_file_path, 'r', encoding='utf-8') as f:
            for sent in conllu.parse_incr(f):
                for node in sent:
                    if not isinstance(node['id'], int):
                        continue
                    dep_id = node['deprel']
                    dep_id = dep_id.split(":")[0]
                    dep_id_set.add(dep_id)

    output_dir = "experiments/vocabs/ud/"
    with open(f'{output_dir}/head_tags.txt', 'x', encoding='utf-8') as f:
        for token in dep_id_set:
            f.write(f'{token}\n')

    with open(f'{output_dir}/non_padded_namespaces.txt', 'x', encoding='utf-8') as f:
        for token in ["*tags", "*labels"]:
            f.write(f'{token}\n')


if __name__ == "__main__":
    copy_standard_datasets()
    create_ud_vocab()
    create_reordered_datasets_script()
