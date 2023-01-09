import os
import pathlib
import random
from typing import Dict, Tuple, List
import logging


from conllu import parse_incr, TokenTree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)

'''
This custom UniversalDependenciesDatasetReader has the following modifications:
1) The read() function can read not only a single conllu file like the original class but also a full directory of 
conllu files and a list of conllu files separated by a comma.
'''

# region UD Utils
def get_tree_ids(tree: TokenTree):
    tree_ids = []
    token_id = tree.token["id"]
    if isinstance(token_id, int):
        tree_ids.append(int(token_id))
    for sub_tree in tree.children:
        sub_tree_ids = get_tree_ids(sub_tree)
        tree_ids.extend(sub_tree_ids)
    return tree_ids


def verify_all_sub_trees(tree: TokenTree):
    tree_ids = set(get_tree_ids(tree))
    expected_ids = set(range(min(tree_ids), min(tree_ids) + len(tree_ids)))
    if not tree_ids == expected_ids:
        raise AssertionError("error")

    for sub_tree in tree.children:
        verify_all_sub_trees(sub_tree)


def is_projective(tree: TokenTree):
    try:
        verify_all_sub_trees(tree)
        return True
    except AssertionError:
        return False


def get_all_tokens(tree: TokenTree):
    tokens = []
    token_id = tree.token["id"]
    if isinstance(token_id, int):
        tokens.append(tree.token)
    for sub_tree in tree.children:
        sub_tree_tokens = get_all_tokens(sub_tree)
        tokens.extend(sub_tree_tokens)
    return tokens


def shift_subtree(tree: TokenTree, new_start_position: int):
    assert new_start_position > 0

    tree_min_id = min(get_tree_ids(tree))
    diff = new_start_position - tree_min_id
    tree_tokens = get_all_tokens(tree)
    for token in tree_tokens:
        token["id"] = token["id"] + diff

    return tree_tokens


def shuffle_ud_tree_recursive(tree: TokenTree):
    verify_all_sub_trees(tree)

    if len(tree.children) == 0:
        return
    for subtree in tree.children:
        shuffle_ud_tree_recursive(subtree)

    subtrees_indexes = list(range(len(tree.children) + 1))
    random.shuffle(subtrees_indexes)
    subtrees_len = list(map(lambda x: len(get_tree_ids(x)), tree.children))
    tree_min_id = min(get_tree_ids(tree))
    new_position_start_id = tree_min_id

    for new_postion, old_position in enumerate(subtrees_indexes):
        # Root node
        if old_position == len(tree.children):
            tree.token["id"] = new_position_start_id
            new_position_start_id = new_position_start_id + 1
        else:
            subtree_len = subtrees_len[old_position]
            shift_subtree(tree.children[old_position], new_position_start_id)
            new_position_start_id = new_position_start_id + subtree_len

    new_tree_ids = get_tree_ids(tree)
    if len(new_tree_ids) != len(set(new_tree_ids)):
        raise Exception

    if not is_projective(tree):
        raise Exception


def shuffle_ud_tree(tree: TokenTree):
    shuffle_ud_tree_recursive(tree)
    shuffle_map = {}
    shuffle_map[0] = 0
    for token in get_all_tokens(tree):
        shuffle_map[token["original_id"]] = token["id"]

    for token in get_all_tokens(tree):
        token["head"] = shuffle_map[token["head"]]


# endregion

# exist_ok has to be true until we remove this from the core library
@DatasetReader.register("universal_dependencies_custom", exist_ok=True)
class UniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : `bool`, optional (default = `False`)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `Tokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        tokenizer: Tokenizer = None,
        max_sentence_length_filter: int = None,
        shuffle_ud_tree: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer
        self.max_sentence_length_filter = max_sentence_length_filter
        self.shuffle_ud_tree = shuffle_ud_tree

    def _read(self, file_path: str):
        if os.path.isdir(file_path):
            file_path_list = list(pathlib.Path(file_path).glob("**/*.conllu"))
        else:
            file_path_list = file_path.split(",")

        for file_path in file_path_list:
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)

            with open(file_path, 'r', encoding="utf-8") as conllu_file:
                logger.info("Reading UD instances from conllu dataset at: %s", file_path)
                for annotation in parse_incr(conllu_file):
                    # CoNLLU annotations sometimes add back in words that have been elided
                    # in the original sentence; we remove these, as we're just predicting
                    # dependencies for the original sentence.
                    # We filter by None here as elided words have a non-integer word id,
                    # and are replaced with None by the conllu python library.

                    if self.shuffle_ud_tree:
                        ud_tree = annotation.to_tree()
                        if is_projective(ud_tree):
                            for token in get_all_tokens(ud_tree):
                                token["original_id"] = token["id"]
                            shuffle_ud_tree(ud_tree)


                    conllu_data = annotation

                    # filter out sub-nodes (ie nodes with id such as '1.2')
                    annotation = [x for x in annotation if isinstance(x["id"], int)]
                    annotation = sorted(annotation, key=lambda x: x["id"])

                    heads = [x["head"] for x in annotation]
                    tags = [x["deprel"].split(":")[0] for x in annotation]
                    words = [x["form"] for x in annotation]

                    if self.max_sentence_length_filter and len(words) > self.max_sentence_length_filter:
                        continue

                    if self.use_language_specific_pos:
                        pos_tags = [x["xpostag"] for x in annotation]
                    else:
                        pos_tags = [x["upostag"] for x in annotation]

                    ids = [x["id"] for x in annotation]

                    yield self.text_to_instance(ids, words, pos_tags, list(zip(tags, heads)), conllu_data)

    def text_to_instance(self,  # type: ignore
                         ids: List[str],
                         words: List[str],
                         upos_tags: List[str],
                         dependencies: List[Tuple[str, int]] = None,
                         metadata=None
                         ) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies : ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]

        text_field = TextField(tokens, self._token_indexers)
        fields["words"] = text_field
        fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     text_field,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([x[1] for x in dependencies],
                                                        text_field,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags, "ids": ids, "metadata": metadata})
        return Instance(fields)