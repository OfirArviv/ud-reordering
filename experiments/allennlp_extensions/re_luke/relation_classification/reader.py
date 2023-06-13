import logging
from random import random
from typing import Dict, Optional
import json
from pathlib import Path
import numpy as np
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance, Token
from allennlp.data.fields import SpanField, TextField, LabelField, TensorField, MetadataField

from ..utils.util import ENT, ENT2, list_rindex
logger = logging.getLogger(__name__)


def parse_tacred_file(path: str):
    if Path(path).suffix != ".json":
        raise ValueError(f"{path} does not seem to be a json file. We currently only supports the json format file.")
    for example in json.load(open(path, "r")):
        tokens = example["token"]
        spans = [
            ((example["subj_start"], ENT), (example["subj_end"] + 1, ENT)),
            ((example["obj_start"], ENT2), (example["obj_end"] + 1, ENT2)),
        ]

        # carefully insert special tokens in a specific order
        spans.sort()
        for i, span in enumerate(spans):
            (start_idx, start_token), (end_idx, end_token) = span
            tokens.insert(end_idx + i * 2, end_token)
            tokens.insert(start_idx + i * 2, start_token)

        sentence = " ".join(tokens)
        # we do not need some spaces
        sentence = sentence.replace(f" {ENT} ", f"{ENT} ")
        sentence = sentence.replace(f" {ENT2} ", f"{ENT2} ")

        yield {"example_id": example["id"], "sentence": sentence, "label": example["relation"]}

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


@DatasetReader.register("relation_classification")
class RelationClassificationReader(DatasetReader):
    def __init__(
        self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], use_entity_feature: bool = False,
            source_max_tokens: Optional[int] = None, max_examples: Optional[int] = None, **kwargs,
    ):
        super().__init__(**kwargs)

        self.parser = parse_conll_file
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.use_entity_feature = use_entity_feature

        self.head_entity_id = 1
        self.tail_entity_id = 2

        self._source_max_tokens = source_max_tokens
        self._source_max_exceeded = 0
        self._max_examples = max_examples

    def text_to_instance(self, sentence: str, label: str = None):
        texts = [t.text for t in self.tokenizer.tokenize(sentence)]
        e1_start_position = texts.index(ENT)
        e1_end_position = list_rindex(texts, ENT)

        e2_start_position = texts.index(ENT2)
        e2_end_position = list_rindex(texts, ENT2)

        tokens = [Token(t) for t in texts]
        text_field = TextField(tokens, token_indexers=self.token_indexers)

        fields = {
            "word_ids": text_field,
            "entity1_span": SpanField(e1_start_position, e1_end_position, text_field),
            "entity2_span": SpanField(e2_start_position, e2_end_position, text_field),
            "input_sentence": MetadataField(sentence),
        }

        if label is not None:
            fields["label"] = LabelField(label)

        if self.use_entity_feature:
            fields["entity_ids"] = TensorField(np.array([self.head_entity_id, self.tail_entity_id]))

        return Instance(fields)

    def _read(self, file_path: str):
        self._processed_examples = 0
        data_list = self.parser(file_path)
        if self._max_examples is not None:
            assert len(data_list) >= self._max_examples
            random.shuffle(data_list)

        for data in data_list:
            if self._max_examples is not None and self._processed_examples > self._max_examples:
                break

            tokenized_source_len = len([t.text for t in self.tokenizer.tokenize(data["sentence"])])
            if self._source_max_tokens and tokenized_source_len > self._source_max_tokens:
                self._source_max_exceeded += 1
                continue

            # if data['label'] not in self.labels:
            #     continue
            self._processed_examples += 1
            yield self.text_to_instance(data["sentence"], data["label"])

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were skipped.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
