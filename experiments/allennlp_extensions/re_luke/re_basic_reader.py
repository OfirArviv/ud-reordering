import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from allennlp.data import Tokenizer

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)
ENT = "<ent>"
ENT2 = "<ent2>"

@DatasetReader.register("re_basic")
class ReBasic(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.

    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. `"5-class"` uses these labels as is. `"3-class"` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). `"2-class"` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).

    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.

    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`

    Registered as a `DatasetReader` with name "sst_tokens".

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    use_subtrees : `bool`, optional, (default = `False`)
        Whether or not to use sentiment-tagged subtrees.
    granularity : `str`, optional (default = `"5-class"`)
        One of `"5-class"`, `"3-class"`, or `"2-class"`, indicating the number
        of sentiment labels to use.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Optional[Tokenizer],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer

    @staticmethod
    def parse_conll_file(path: str):
        if Path(path).suffix != ".json":
            raise ValueError(
                f"{path} does not seem to be a json file. We currently only supports the json format file.")
        for example in json.load(open(path, "r")):
            tokens = example["tokens"]

            for rel in example['relations']:
                ent_1_idx = int(rel['head'])
                ent_2_idx = int(rel['tail'])
                ent1 = example["entities"][ent_1_idx]
                ent2 = example["entities"][ent_2_idx]
                spans = [
                    ((ent1['start'], ENT), (ent1['end'], ENT)),
                    ((ent2['start'], ENT2), (ent2['end'], ENT2))
                ]

                # carefully insert special tokens in a specific order
                spans.sort()
                for i, span in enumerate(spans):
                    (start_idx, start_token), (end_idx, end_token) = span
                    tokens.insert(end_idx + i * 2, end_token)
                    tokens.insert(start_idx + i * 2, start_token)

                sentence = " ".join(tokens)
                # we do not need some spaces
                # sentence = sentence.replace(f" {ENT} ", f"{ENT} ")
                # sentence = sentence.replace(f" {ENT2} ", f"{ENT2} ")

                yield {"example_id": example["orig_id"], "sentence": sentence, "label": rel["type"]}

    def _read(self, file_path: str):
        for data in self.parse_conll_file(file_path):
            if len([t.text for t in self._tokenizer.tokenize(data["sentence"])]) > 512:
                continue
            yield self.text_to_instance(data["sentence"].split(), data["label"])

    def text_to_instance(self, tokens: List[str], sentiment: str = None) -> Optional[Instance]:
        """
        We take `pre-tokenized` input here, because we might not have a tokenizer in this class.

        # Parameters

        tokens : `List[str]`, required.
            The tokens in a given sentence.
        sentiment : `str`, optional, (default = `None`).
            The sentiment for this sentence.

        # Returns

        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence or phrase.
            label : `LabelField`
                The sentiment label of the sentence or phrase.
        """
        tokens = self._tokenizer.tokenize(" ".join(tokens))
        text_field = TextField(tokens)
        fields: Dict[str, Field] = {"tokens": text_field}
        if sentiment is not None:
            fields["label"] = LabelField(sentiment)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self._token_indexers
