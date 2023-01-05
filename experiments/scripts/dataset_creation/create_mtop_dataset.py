import glob
import json
import os
import random
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Union
import nltk
from tqdm import tqdm

from reordering_package.ud_reorder_algo import UdReorderingAlgo


# region Legacy Code

def _parse_tree_to_source_sequence(parse_tree: nltk.Tree) -> str:
    tree_str = " ".join(parse_tree.leaves()).replace("(", "[").replace(")", "]").replace("{", "(").replace("}", ")") \
        .replace("\n ", "").replace("]", " ]")

    return tree_str


# TODO: There is about 10% error rate. Most are 'error:inconsistent span exist'
def reorder_mtop(parse_tree: nltk.Tree, reorder_algo: UdReorderingAlgo, reorder_by_lang: str) -> nltk.Tree:
    parse_tree = parse_tree.copy(deep=True)
    sentence = _parse_tree_to_source_sequence(parse_tree)
    # TODO: Currently we are not using the entity-aware reordering and just filter ill-ordered instances.
    try:
        mapping = reorder_algo.get_entities_aware_reorder_mapping(sentence, reorder_by_lang, [])
    except Exception as e:
        raise Exception(f'Failed to reorder. {e}')

    if mapping is None:
        raise Exception("Can not reorder sentence.")

    reordered_sentence = reorder_algo.reorder_sentence(sentence, mapping)

    inconsistent_span_exist = False

    def traverse(t, source_sequence: str, start_idx_in_og_string: int) -> None:
        try:
            t.label()
        except AttributeError:
            return
        span_text_list = [[c] if isinstance(c, str) else c.leaves() for c in t]
        if len(span_text_list) > 1:
            source_sequence_arr = source_sequence.replace("(", "{").replace(")", "}").split(" ")
            og_span_idx_list = []
            span_idx_list = []
            idx = start_idx_in_og_string
            for span_text in span_text_list:
                start_idx = idx
                end_idx = idx + len(span_text) - 1
                assert source_sequence_arr[start_idx:end_idx + 1] == span_text
                idx = end_idx + 1

                span_idx = (start_idx, end_idx)
                og_span_idx_list.append(span_idx)
                res = UdReorderingAlgo.get_continuous_mapped_span(span_idx, mapping)
                if res is None:
                    nonlocal inconsistent_span_exist
                    inconsistent_span_exist = True
                    span_idx_list.append(span_idx)
                else:
                    span_idx_list.append(res)

            span_idx_dict = {k: v for k, v in enumerate(span_idx_list)}
            sorted_span_idx_dict = {k: v for k, v in sorted(span_idx_dict.items(), key=lambda item: item[1][0])}
            subtree_nodes = [c for c in t]
            reordered_subtree_nodes = [subtree_nodes[i] for i in sorted_span_idx_dict.keys()]

            reordered_og_span_idx_list = [og_span_idx_list[i] for i in sorted_span_idx_dict.keys()]

            for c in subtree_nodes:
                t.remove(c)
            t.extend(reordered_subtree_nodes)

            child_idx = 0
            for child in t:
                if not isinstance(child, str):
                    start_idx_in_og_string = reordered_og_span_idx_list[child_idx][0]
                    traverse(child, source_sequence, start_idx_in_og_string)
                child_idx = child_idx + 1
        else:
            child_idx = 0
            for child in t:
                if not isinstance(child, str):
                    traverse(child, source_sequence, start_idx_in_og_string)
                child_idx = child_idx + 1

    traverse(parse_tree, sentence, 0)

    if inconsistent_span_exist:
        raise Exception("inconsistent span exist")

    assert reordered_sentence == _parse_tree_to_source_sequence(
        parse_tree), f'failed to reorder tree. {reordered_sentence} != {_parse_tree_to_source_sequence(parse_tree)}'

    return parse_tree


def full_parse_tree_to_decoupled_format(parse_tree: nltk.Tree) -> nltk.Tree:
    decoupled_tree = parse_tree.copy(True)

    def traverse(t) -> None:
        try:
            t.label()
        except AttributeError:
            return

        if not t.label().startswith("SL"):
            text_leaves = [c for c in t if isinstance(c, str)]
            for c in text_leaves:
                t.remove(c)

        for child in t:
            traverse(child)

    traverse(decoupled_tree)

    return decoupled_tree


def full_parse_tree_to_pointer_format(parse_tree: nltk.Tree) -> nltk.Tree:
    parse_tree = parse_tree.copy(True)

    def traverse(t, idx: int) -> int:
        try:
            t.label()
        except AttributeError:
            return idx

        for i, child in enumerate(t):
            if isinstance(child, str):
                t[i] = f'@ptr{idx}'
                idx = idx + 1
            else:
                idx = traverse(child, idx)
        return idx

    traverse(parse_tree, 0)

    return parse_tree


def get_seq2seq_str_from_pointer_parse_tree(parse_tree: nltk.Tree,
                                            use_pointers: bool,
                                            use_decoupled_format: bool) -> str:
    source_sequence = _parse_tree_to_source_sequence(parse_tree)

    if use_pointers:
        parse_tree = full_parse_tree_to_pointer_format(parse_tree)
    if use_decoupled_format:
        parse_tree = full_parse_tree_to_decoupled_format(parse_tree)

    target_sequence = _parse_tree_to_parse_str(parse_tree)

    string = f'{source_sequence}\t{target_sequence}'
    return string


def _partial_format_tree_to_full_format_tree(partial_format_tree: nltk.Tree, source_sequence: str) -> nltk.Tree:
    parse_tree = partial_format_tree
    children = [c for c in parse_tree]
    children_str = [c if isinstance(c, str) else " ".join(c.leaves()) for c in parse_tree]
    source_sequence_arr = source_sequence.replace("(", "{").replace(")", "}").split(" ")

    children_span_idx = []
    for c in children_str:
        span_start = _getsubidx(source_sequence_arr, c.split())
        span_end = span_start + len(c.split(" ")) - 1
        children_span_idx.append(span_start)

    children = [x for _, x in sorted(zip(children_span_idx, children))]
    children_str = [x for _, x in sorted(zip(children_span_idx, children_str))]

    if len(children) == 0:
        new_children_arr = source_sequence_arr
    else:
        new_children_arr = []

        first_child_idx = _getsubidx(source_sequence_arr, children_str[0].split(" "))
        if first_child_idx > 0:
            extra_str = source_sequence_arr[:first_child_idx]
            for s in extra_str:
                new_children_arr.append(s)

        if len(children) == 1:
            new_children_arr.append(children[0])

        idx = 0
        while idx + 1 < len(children_str):
            first_child = children_str[idx]
            first_child_idx = _getsubidx(source_sequence_arr, first_child.split(" "))
            first_child_end_idx = first_child_idx + len(first_child.split(" ")) - 1
            second_child = children_str[idx + 1]
            second_child_idx = _getsubidx(source_sequence_arr, second_child.split(" "))
            if len(new_children_arr) == 0 or new_children_arr[-1] != children[idx]:
                new_children_arr.append(children[idx])
            if first_child_end_idx + 1 < second_child_idx:
                missing_str = source_sequence_arr[first_child_end_idx + 1:second_child_idx]
                new_children_arr.extend(missing_str)
            new_children_arr.append(children[idx + 1])

            idx = idx + 1

        last_char_index = _getsubidx(source_sequence_arr, children_str[-1].split(" ")) + len(
            children_str[-1].split(" ")) - 1
        if last_char_index + 1 < len(source_sequence_arr):
            extra_str = source_sequence_arr[last_char_index + 1:]
            for s in extra_str:
                new_children_arr.append(s)

    temp_parse_tree: nltk.Tree = parse_tree.copy(deep=True)
    temp_parse_tree.clear()
    temp_parse_tree.extend(new_children_arr)
    full_parse_tree = temp_parse_tree

    if _parse_tree_to_source_sequence(full_parse_tree) != source_sequence:
        assert False, "converting to full tree format failed"

    return full_parse_tree


# TODO: Add non-pointer instance
def create_mtop_seq2seq_dataset_legacy(input_file_path: str,
                                       input_lang: str,
                                       use_decoupled_format: bool,
                                       use_pointer_format: bool,
                                       reorder_algo: UdReorderingAlgo.ReorderAlgo,
                                       reorder_by_lang: Optional[str] = None):
    ud_reorder_algo = UdReorderingAlgo(reorder_algo, input_lang)
    seq2seq_std_strings = []
    seq2seq_reordered_strings = []
    with open(input_file_path, "r", encoding='utf-8') as input_file:
        error = 0
        total = 0
        for line_num, line in tqdm(enumerate(input_file)):
            total += 1
            line = line.strip("\n")
            if not line:
                continue
            line_parts = line.split("\t")
            _id, intent, slot_string, utterance, domain, locale, decoupled_form_string, token_json = line_parts
            source_sequence = " ".join(json.loads(token_json)['tokens'])
            partial_parse_tree = _get_parse_tree_from_str(decoupled_form_string)
            try:
                parse_tree = _partial_format_tree_to_full_format_tree(partial_parse_tree, source_sequence)
                if reorder_by_lang:
                    reordered_parse_tree = reorder_mtop(parse_tree, ud_reorder_algo, reorder_by_lang)
            except Exception as e:
                # print(f'error:{str(e)}')
                error += 1
                continue

            if use_decoupled_format:
                if not full_parse_tree_to_decoupled_format(parse_tree) == partial_parse_tree:
                    print("here")

            # seq2seq_std_strings.append(get_seq2seq_str(parse_tree, use_pointer_format, use_decoupled_format))
            # if reorder_by_lang:
            #    seq2seq_reordered_strings.append(get_seq2seq_str(reordered_parse_tree, use_pointer_format,
            #                                                     use_decoupled_format))

    output_dir = "experiments/datasets/top/seq2seq_mtop/"
    filename = os.path.basename(input_file_path).split(".")[0]

    # print the non-combined file. Either the std or the reordered
    output_file_path = f'{output_dir}/{filename}_0' \
                       f'{"_reordered_by_" + reorder_by_lang + "_" + reorder_algo.name if reorder_by_lang is not None else ""}' \
                       f'{"_decoupled_v2" if decoupled_form_string is not None else ""}' \
                       '.tsv'

    with open(output_file_path, 'x', encoding='utf-8') as f:
        seq2seq_strings = seq2seq_reordered_strings if reorder_by_lang else seq2seq_std_strings
        for s in seq2seq_strings:
            f.write(f'{s}\n')


# endregion


# region Tree Utils

def _convert_bare_closing_parentesis_to_full(decoupled_form_string: str):
    def _find_parens(s):
        toret = {}
        pstack = []

        for i, c in enumerate(s.split(" ")):
            if c.startswith('['):
                pstack.append(i)
            elif c == ']':
                if len(pstack) == 0:
                    raise IndexError("No matching closing parens at: " + str(i))
                toret[pstack.pop()] = i

        if len(pstack) > 0:
            raise IndexError("No matching opening parens at: " + str(pstack.pop()))

        return toret

    matching_parentheses_dict = _find_parens(decoupled_form_string)
    decoupled_form_string_formatted = decoupled_form_string.split(" ")
    for s_idx in matching_parentheses_dict.keys():
        e_idx = matching_parentheses_dict[s_idx]
        s_idx_type = decoupled_form_string_formatted[s_idx][1:]
        e_idx_new_str = f'{s_idx_type}]'
        decoupled_form_string_formatted[e_idx] = e_idx_new_str
    target_sequence = " ".join(decoupled_form_string_formatted)

    return target_sequence


def _get_parse_tree_from_str(tree_string: str) -> nltk.Tree:
    temp_tree_str = tree_string.replace("(", "{").replace(")", "}").split(" ")
    tree_str = []
    for i in temp_tree_str:
        if i.startswith('['):
            tree_str.append("(")
            tree_str.append(i[1:])
        elif i == "]":
            tree_str.append(")")
        else:
            tree_str.append(i)

    parse_tree = nltk.Tree.fromstring(" ".join(tree_str))

    return parse_tree


def _getsubidx(lst: List, sublist: List) -> Optional[int]:
    # This function return the index of the first occurrence of a sublist in a containing list
    l1, l2 = len(lst), len(sublist)
    for i in range(l1):
        if lst[i:i + l2] == sublist:
            return i


def _get_sub_sentence_tokens_positions(sentence: List, sub_sentence: List) -> List[int]:
    allowed_splits_count = [2]
    for split_count in allowed_splits_count:
        for i in range(1, len(sub_sentence)):
            first_half = sub_sentence[:i]
            second_half = sub_sentence[i:]
            first_half_start = _getsubidx(sentence, first_half)
            second_half_start = _getsubidx(sentence, second_half)

            if first_half_start is not None and second_half_start is not None:
                first_half_indexes = list(range(first_half_start, first_half_start + len(sub_sentence[:i])))
                second_half_indexes = list(range(second_half_start, second_half_start + len(sub_sentence[i:])))

                return first_half_indexes + second_half_indexes


def parse_tree_to_pointer_format(parse_tree: nltk.Tree, source_sequence: str) -> nltk.Tree:
    parse_tree = parse_tree.copy(True)

    def traverse(t) -> None:
        try:
            t.label()
        except AttributeError:
            # it's a string leaf
            return

        span_idx_list = []
        curr_span_idx = 0
        span_idx_list.append([])
        for i, child in enumerate(t):
            if isinstance(child, str):
                span_idx_list[curr_span_idx].append(i)
            else:
                curr_span_idx = curr_span_idx + 1
                span_idx_list.append([])
        for span in span_idx_list:
            span_txt = [t[i] for i in span]
            span_txt = list(map(lambda x: x.replace("{", "(").replace("}", ")"), span_txt))
            span_start = _getsubidx(source_sequence.split(), span_txt)
            curr_idx = span_start
            if curr_idx is None:
                heuristic_list = _get_sub_sentence_tokens_positions(source_sequence.split(), span_txt)
                if heuristic_list is None:
                    raise ValueError("parse_tree_to_pointer_format: Can't find the span index. "
                                     "The span probably is missing a token")
                else:
                    heuristic_list = sorted(heuristic_list)
                    for i, j in zip(span, heuristic_list):
                        t[i] = j
            else:
                for i in span:
                    t[i] = curr_idx
                    curr_idx = curr_idx + 1

        for child in t:
            traverse(child)

    traverse(parse_tree)

    return parse_tree


def _parse_tree_to_parse_str(parse_tree: nltk.Tree) -> str:
    tree_str = str(parse_tree).replace("(", "[").replace(")", "]").replace("{", "(").replace("}", ")") \
        .replace("\n ", "").replace("]", " ]")
    assert "\n" not in tree_str
    assert "[[" not in tree_str

    return _convert_bare_closing_parentesis_to_full(tree_str)


def get_parse_tree_str_from_pointer_tree(parse_tree: nltk.Tree, source_sequence: str, use_pointers: bool) -> str:
    parse_tree = parse_tree.copy(True)
    source_sequence_list = source_sequence.split()

    def traverse_replace_idx(t):
        try:
            t.label()
        except AttributeError:
            return

        for i, child in enumerate(t):
            if isinstance(child, int):
                if use_pointers:
                    t[i] = f'@ptr{t[i]}'
                else:
                    t[i] = source_sequence_list[t[i]]
            else:
                traverse_replace_idx(child)

    traverse_replace_idx(parse_tree)

    target_sequence = _parse_tree_to_parse_str(parse_tree)

    return target_sequence


# endregion


# region Pointer Tree Reordering Funcs

def get_spans_from_tree(parse_tree: nltk.Tree,
                        is_decoupled_form: bool) -> List[Dict]:
    # In decoupled form, the order of the text leaves does not must follow the word order.
    # Thus, we only care about the spans in the lowest subtrees. We assume that in a subtree
    # there is only  single span. i.e. There is no case of [w1 w2 T[w3] w4], as in this case we will extract
    # the span w1 w2 w4, which can never be continuous.

    parse_tree: nltk.Tree = parse_tree.copy(True)
    span_list = []

    def traverse(t: nltk.Tree) -> None:
        try:
            t.label()
        except AttributeError:
            # it's a string leaf
            return

        if is_decoupled_form:
            leaves = [child for child in t if isinstance(child, int) or isinstance(child, str)]
        else:
            leaves = [[c] if isinstance(c, str) else c.leaves() for c in t]
            leaves = [item for sublist in leaves for item in sublist]

        nonlocal span_list
        if len(leaves) > 0:
            span_list.append(leaves)

        for child in t:
            traverse(child)

    traverse(parse_tree)

    inclusive_spans_dict_list = []
    for span in span_list:
        start, end = min(span), max(span)
        assert set(range(start, end + 1)) == set(span), "In: get_spans_from_tree() - non continuous span."
        inclusive_spans_dict_list.append({"inclusive_span": (start, end)})

    return inclusive_spans_dict_list


def reorder_mtop_pointer_tree(parse_tree: nltk.Tree,
                              reorder_algo: UdReorderingAlgo,
                              source_sequence: str,
                              reorder_by_lang: str,
                              is_decoupled_form) -> (str, nltk.Tree):
    parse_tree: nltk.Tree = parse_tree.copy(deep=True)
    tree_spans = get_spans_from_tree(parse_tree, is_decoupled_form)
    try:
        mapping = reorder_algo.get_entities_aware_reorder_mapping(source_sequence, reorder_by_lang, tree_spans)
    except Exception as e:
        raise Exception(f'Failed to reorder. {e}')

    if mapping is None:
        raise Exception("Can not reorder sentence.")

    reordered_sentence = reorder_algo.reorder_sentence(source_sequence, mapping)

    # First step: Replace the leaves in the tree (the pointers) according to the reorder mapping
    def traverse_replace_pointers(t: nltk.Tree) -> None:
        try:
            t.label()
        except AttributeError:
            return

        # We are first traversing into the subtrees, as otherwise the dynamic change
        # of the tree mid-iteration, cause errors
        for c in t:
            if not isinstance(c, int):
                traverse_replace_pointers(c)

        mapped_children = []
        for c in t:
            if isinstance(c, int):
                # The mapping indexes start from 1 and the pointers start from 0.
                mapped_node = mapping[c + 1] - 1
                mapped_children.append(mapped_node)
            else:
                mapped_children.append(c)

        for c in list(t):
            t.remove(c)
        t.extend(mapped_children)

    mapped_parse_tree: nltk.Tree = parse_tree.copy(deep=True)
    traverse_replace_pointers(mapped_parse_tree)

    # Second step: Reorder the nodes of the tree according to the new indexes
    def traverse_reorder_nodes(t: nltk.Tree) -> None:
        try:
            t.label()
        except AttributeError:
            return

        if len(t.leaves()) == 0:
            return

        # We are first traversing into the subtrees, as otherwise the dynamic change
        # of the tree mid-iteration, cause errors
        for c in t:
            if not isinstance(c, int):
                traverse_reorder_nodes(c)

        subtree_start_index_to_subtree = dict()
        for c in t:
            if isinstance(c, int):
                subtree_start_index_to_subtree[c] = c
            else:
                subtree_leaves = c.leaves()

                if len(subtree_leaves) == 0:
                    current_loc = subtree_start_index_to_subtree.keys()
                    if len(current_loc) == 0:
                        current_loc = [-1]
                    subtree_start_index = min(current_loc) + 0.5
                else:
                    subtree_start_index = min(subtree_leaves)
                subtree_start_index_to_subtree[subtree_start_index] = c

        for c in list(t):
            t.remove(c)

        for k in sorted(subtree_start_index_to_subtree.keys()):
            t.append(subtree_start_index_to_subtree[k])

    reordered_parse_tree: nltk.Tree = mapped_parse_tree.copy(deep=True)
    traverse_reorder_nodes(reordered_parse_tree)
    # Third Step: Validate the tree has only continuous spans
    # This function also validate the spans are continuous
    try:
        get_spans_from_tree(reordered_parse_tree, is_decoupled_form)
    except Exception:
        raise ValueError("Non continuous span in reordered tree!")

    return reordered_sentence, reordered_parse_tree


# endregion


# region Dataset Creation
def create_mtop_seq2seq_dataset(input_file_path: Union[str, Tuple[str, str]],
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
    with open(input_file_path, "r", encoding='utf-8') as input_file:
        for line_num, line in tqdm(enumerate(input_file)):
            line = line.strip("\n")
            if not line:
                continue

            line_parts = line.split("\t")
            _id, intent, slot_string, utterance, domain, locale, decoupled_form_string, token_json = line_parts
            source_sequence = " ".join(json.loads(token_json)['tokens'])

            parse_tree = _get_parse_tree_from_str(decoupled_form_string)
            try:
                pointers_parse_tree = parse_tree_to_pointer_format(parse_tree, source_sequence)
            except Exception as e:
                error_dict[f'error converting to pointer format - {str(e)}'] += 1
                # print(e)
                continue

            if not use_decoupled_format:
                raise NotImplemented()
                # TODO: Add function that uses pointers format
                # parse_tree = _partial_format_tree_to_full_format_tree(parse_tree, source_sequence)

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
                    # print(e)
                    seq2seq_reordered_strings.append(f'{source_sequence}\t{parse_tree_str}\n')

            parse_tree_str = get_parse_tree_str_from_pointer_tree(pointers_parse_tree,
                                                                  source_sequence,
                                                                  use_pointer_format)
            seq2seq_std_strings.append(f'{source_sequence}\t{parse_tree_str}\n')

    print(error_dict)

    filename = os.path.basename(input_file_path).split(".")[0]

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


def create_english_dataset_script_mtop(use_pointers: bool):
    use_decoupled_format = True
    output_dir = "experiments/processed_datasets/mtop/"
    if use_pointers:
        output_dir += "pointers_format/"
    output_dir += "/standard/"
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'eval', 'test']:
        print(f'Creating seq2seq dataset. lang: en, split: {split}')
        create_mtop_seq2seq_dataset(f'experiments/datasets/top/mtop/en/{split}.txt',
                                    "english",
                                    use_pointers,
                                    use_decoupled_format,
                                    output_dir,
                                    None,
                                    None)


def create_test_datasets_script_mtop(use_pointers: bool):
    use_decoupled_format = True
    output_dir = "experiments/processed_datasets/mtop/"
    if use_pointers:
        output_dir += "pointers_format/"
    output_dir += "/standard/"
    os.makedirs(output_dir, exist_ok=True)
    for lang in ['hi', 'th', 'de', 'es', 'fr']:
        print(f'Creating seq2seq dataset. lang: {lang}, split: test')
        create_mtop_seq2seq_dataset(f'experiments/datasets/top/mtop/{lang}/test.txt',
                                    lang,
                                    use_pointers,
                                    use_decoupled_format,
                                    output_dir,
                                    None,
                                    None)


def create_reordered_datasets_script_mtop(use_pointers: bool):
    use_decoupled_format = True
    for lang in ['hindi', 'thai', 'french', 'spanish', 'german']:
        output_dir = f'experiments/processed_datasets/mtop/'
        if use_pointers:
            output_dir += "pointers_format/"
        output_dir += f'english_reordered_by_{lang}'
        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'eval', 'test']:
            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(f'Creating seq2seq dataset. lang: {lang}, split: {split}, algorithm: {reorder_algo.name}')
                create_mtop_seq2seq_dataset(f'experiments/datasets/top/mtop/en/{split}.txt',
                                            "english",
                                            use_pointers,
                                            use_decoupled_format,
                                            output_dir,
                                            reorder_algo,
                                            lang)


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
                    if token.startswith('[') or token.startswith(']') or token.endswith(']'):
                        if token not in ontology_tokens:
                            ontology_tokens.append(token)
                    elif token.startswith('@ptr'):
                        pointers_tokens.add(token)
                    else:
                        raise NotImplemented(token)

    with open(f'{output_dir}/target_tokens.txt', 'w', encoding='utf-8') as f:
        for token in ontology_tokens:
            f.write(f'{token}\n')

        pointer_tokens_size = 100
        assert pointer_tokens_size > len(pointers_tokens)
        for i in range(pointer_tokens_size):
            f.write(f'@ptr{i}\n')

    with open(f'{output_dir}/non_padded_namespaces.txt', 'w', encoding='utf-8') as f:
        for token in ["*tags", "*labels"]:
            f.write(f'{token}\n')


def convert_pointer_format_tsv_to_non_pointer_format(pointer_format_dataset_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    subdirs = list(os.walk(pointer_format_dataset_dir))[0][1]
    for d in subdirs:
        input_dir = f'{pointer_format_dataset_dir}/{d}'
        output_subdir = f'{output_dir}/{d}'
        os.makedirs(output_subdir, exist_ok=True)
        file_paths = glob.glob(f'{input_dir}/*.tsv')

        for file_path in file_paths:
            file_basename = os.path.basename(file_path)
            output_file_path = f'{output_subdir}/{file_basename}'
            with open(file_path, 'r', encoding='utf-8') as f, \
                    open(output_file_path, 'x', encoding='utf-8') as o_f:
                for line in f:
                    error = False
                    source_seq, target_seq = line.strip("\n").split("\t")
                    source_seq_arr = source_seq.split()
                    non_pointer_target_seq_arr = []
                    for tok in target_seq.split():
                        if tok.startswith("@ptr"):
                            pointer_idx = int(tok.split("@ptr")[1])
                            try:
                                non_pointer_target_seq_arr.append(source_seq_arr[pointer_idx])
                            except:
                                error = True
                        else:
                            non_pointer_target_seq_arr.append(tok)
                    if error:
                        continue
                    non_pointer_target_seq = " ".join(non_pointer_target_seq_arr)
                    o_f.write(f'{source_seq}\t{non_pointer_target_seq}\n')

def create_rasoolini_huji_combined_datasets():
    pointer_dataset_path = 'experiments/processed_datasets/mtop/pointers_format/'
    for lang in ['french', 'german', 'hindi', 'spanish', 'thai']:
        for split in ['eval', 'train', 'test']:
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

# endregion



if __name__ == "__main__":
    # TODO: Also run the experiments of the non-decoupled mode to measure if it really help to generalize over
    #  word-order

    create_rasoolini_huji_combined_datasets()
    exit()

    ### MTOP Dataset Creation ###
    # Create English standard dataset
    create_english_dataset_script_mtop(True)

    # Create Reordered Dataset
    create_reordered_datasets_script_mtop(True)

    # Create Test Dataset
    create_test_datasets_script_mtop(True)

    # Create vocab
    create_vocab_from_seq2seq_file([
        "experiments/processed_datasets/mtop/pointers_format/standard/english_train_decoupled_format.tsv",
        "experiments/processed_datasets/mtop/pointers_format/standard/english_eval_decoupled_format.tsv",
        "experiments/processed_datasets/mtop/pointers_format/standard/english_test_decoupled_format.tsv"
    ],
        "experiments/vocabs/mtop_pointers/"
    )

    create_rasoolini_huji_combined_datasets()

    # Create non-pointer dataset
    convert_pointer_format_tsv_to_non_pointer_format("experiments/processed_datasets/mtop/pointers_format/",
                                                     "experiments/processed_datasets/mtop/non_pointer_format/")
