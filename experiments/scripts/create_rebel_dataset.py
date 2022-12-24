import ast
import copy
import glob
import json
import os.path
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator, Optional
import trankit
import pandas as pd
import tqdm
from reordering_package.ud_reorder_algo import UdReorderingAlgo

# region Data Structures
"""
Our annotation pipeline consists of a series of BasePipeline objects.   (EntityLinker, Coreference, triple aligner .. etc)
each BasePipelie class takes a document class and add it's annotation
Each Document with it's annotation when converted into json has the following fields.

  {
        "doc"id:                       Document id     -- Wikipedia document id when dealing with wikipedia dump
        "title":                    title of the wikipedia document
        "uri":                      URI of the item containing the main page
        "text":                     The whole text of the document
        "sentences_boundaries":                start and end offsets of sentences
                                    [(start,end),(start,end)] start/ end are character indices
        "words_boundaries":                                      # list of tuples (start, end) of each word in Wikipedia Article, start/ end are character indices
        "entities":                                             # list of Entities   (Class Entity)
                                    [
                                    {
                                    "uri":
                                    "boundaries": (start,end)   # tuple containing the of the surface form of the entity
                                    "surface-form": ""
                                    "annotator" : ""            # the annotator name used to detect this entity [NER,DBpediaspotlight,coref]
                                    }
                                    ]
        "triples":                  list of triples that occur in the document
                                    We opt of having them exclusive of other fields so they can be self contained and easy to process
                                    [
                                    {
                                    "subject":          class Entity
                                    "predicate":        class Entity
                                    "object":           class Entity
                                    "dependency_path": "lexicalized dependency path between sub and obj if exists" or None (if not existing)
                                    "confidence":      # confidence of annotation if possible
                                    "annotator":       # annotator used to annotate this triple with the sentence
                                    "sentence_id":     # integer shows which sentence does this triple lie in
                                    }
                                    ]
    }
"""


class Document:

    def __init__(self, nlp: trankit.Pipeline, lang: str, docid, title, pageuri, text, sentence_boundaries=None,
                 words_boundaries=None, entities=None, triples=None):
        """

        initalization of document class
        :param id: wikipedia id of each page  # document id if another text dataset is used.
        :param title: title of the page
        :param pageuri: URI of the item containing the main page
        :param text:  "text that is contained in the page"
        :param sentence_boundaries: start and end offsets of sentences
        :param word_boundaries: list of tuples (start, end) of each word in Wikipedia Article, start/ end are character indices
        :param entities: list of Entities in the document
        :param triples:  list of Triples aligned with sentences in the document
        """
        self.nlp = nlp
        if lang != "thai":
            if self.nlp.active_lang != lang:
                self.nlp.set_active(lang)

        self.docid = docid
        self.title = title
        self.uri = pageuri
        self.text = text
        self.sentences_boundaries = self.__get_sentences_boundaries() if sentence_boundaries is None else sentence_boundaries
        self.words_boundaries = self.__get_words_boundaries() if words_boundaries is None else words_boundaries
        self.entities = [] if entities is None else entities
        self.triples = [] if triples is None else triples

    @classmethod
    def fromJSON(cls, j, nlp: trankit.Pipeline, lang: str):
        """
        instantiate a document class from existing json file
        :param j: j is a json file containing all fields as described in the begining of the document
        """

        docid = j['docid']
        title = j['title']
        uri = j['uri']
        text = j['text']
        sentences_boundaries = j['sentences_boundaries'] if 'sentences_boundaries' in j else None
        word_boundaries = j['words_boundaries'] if 'words_boundaries' in j else None
        entities = [Entity.fromJSON(ej) for ej in j['entities']] if 'entities' in j else None
        triples = [Triple.fromJSON(tj) for tj in j['triples']] if 'triples' in j else None

        return Document(nlp, lang, docid, title, uri, text, sentences_boundaries, word_boundaries, entities, triples)

    def __get_sentences_boundaries(self):
        """
        function to tokenize sentences and return
        sentence boundaries of each sentence using a tokenizer.
        :return:
        """
        doc = self.nlp.tokenize(self.text)

        sentences_spans = [sent['dspan'] for sent in doc['sentences']]

        return sentences_spans

    def __get_words_boundaries(self):
        """
        function to tokenize words in the document and return words
        boundaries of each sentence using a tokenizer.
        :return:
        """
        doc = self.nlp.tokenize(self.text)

        words_spans = [tok['dspan'] for sent in doc['sentences'] for tok in sent['tokens']]
        return words_spans

    def toJSON(self):
        """
        function to print the annotated document into one json file
        :return:
        """
        j = self.__dict__.copy()
        j['entities'] = [i.toJSON() for i in j['entities']] if 'entities' in j and j['entities'] is not None else []
        j['triples'] = [i.toJSON() for i in j['triples']] if 'triples' in j and j['triples'] is not None else []
        del j['sentences_boundaries']
        del j['words_boundaries']
        return j

    def get_sentences(self):
        """
        :return: get sentences text
        """
        return [self.text[s:e] for s, e in self.sentences_boundaries]


class Entity:
    def __init__(self, uri, boundaries, surfaceform, annotator=None, type_placeholder=None, property_placeholder=None):
        """
        :param uri: entity uri
        :param boundaries: start and end boundaries of the surface form in the sentence
        :param surfaceform: text containing the surface form of the entity
        :param annotator:   annotator used in entity linking
        """
        self.uri = uri
        self.boundaries = boundaries
        self.surfaceform = surfaceform
        self.annotator = annotator
        # self.type_placeholder = type_placeholder
        # self.property_placeholder = property_placeholder

    @classmethod
    def fromJSON(cls, j):
        """
        initialize an entity class using a json object
        :param j: json object of an entity
        :return: Entity instantiated object
        """
        annotator = j['annotator'] if 'annotator' in j else None
        type_placeholder = j['type_placeholder'] if 'type_placeholder' in j else None
        property_placeholder = j['property_placeholder'] if 'property_placeholder' in j else None
        return Entity(j['uri'], j['boundaries'], j['surfaceform'], annotator, type_placeholder, property_placeholder)

    def toJSON(self):
        return self.__dict__.copy()


class Triple:
    def __init__(self, subject, predicate, object, sentence_id, dependency_path=None, confidence=None, annotator=None):
        """
        :param subject: entity class containing the triple subject
        :param predicate: entity class containing the triple predicate
        :param object:    entity class containing the triple object
        :param sentence_id:  integer showing which sentence in the document this (0,1,2,3 .. first , second , third ..etc)
        :param dependency_path: "lexicalized dependency path between sub and obj if exists" or None (if not existing)
        :param confidence: confidence of annotation if possible
        :param annotator:
        """

        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.sentence_id = sentence_id
        self.dependency_path = dependency_path
        self.confidence = confidence
        self.annotator = annotator

    @classmethod
    def fromJSON(cls, j):
        """
        initialize a triple class using a json object
        :param j: json object of an entity
        :return: Triple instantiated object
        """
        subject = Entity.fromJSON(j['subject'])
        predicate = Entity.fromJSON(j['predicate'])
        object = Entity.fromJSON(j['object'])
        sentence_id = j['sentence_id']
        dependency_path = j['dependency_path'] if 'dependency_path' in j else None
        confidence = j['confidence'] if 'confidence' in j else None
        annotator = j['annotator'] if 'annotator' in j else None

        return Triple(subject, predicate, object, sentence_id, dependency_path, confidence, annotator)

    def toJSON(self):
        j = self.__dict__.copy()
        j['subject'] = j['subject'].toJSON()
        j['predicate'] = j['predicate'].toJSON()
        j['object'] = j['object'].toJSON()

        json_str = json.dumps(j)

        return json_str

    def __repr__(self):
        return json.dumps(self.toJSON())

    def __str__(self):
        return self.__repr__()


# endregion


# region Parsing Utils
TripletsBySubjectBySentence = Dict[str, Dict[str, List[Triple]]]


def _find_sentence_id_by_token_boundaries(sentence_boundaries: List[Tuple[int, int]],
                                          token_boundaries: Tuple[int, int]) -> int:
    tok_start, tok_end = token_boundaries
    for i, (sent_start, sent_end) in enumerate(sentence_boundaries):
        if sent_start <= tok_start < tok_end <= sent_end:
            return i

    raise IndexError()


def _group_triples_by_subject(triples: List[Triple]) -> Dict[str, List[Triple]]:
    triples_by_subject_dict: Dict[str, List[Triple]] = {str(triple.subject.boundaries): [] for triple in triples}

    for triple in triples:
        k = str(triple.subject.boundaries)
        triples_by_subject_dict[k].append(triple)

    return triples_by_subject_dict


def _filter_cross_sentence_triples(triples: List[Triple], sentence_boundaries: List[Tuple[int, int]]) -> List[Triple]:
    filtered_list = []

    for triple in triples:
        try:
            subject_sent_idx = _find_sentence_id_by_token_boundaries(sentence_boundaries, triple.subject.boundaries)
            object_sent_index = _find_sentence_id_by_token_boundaries(sentence_boundaries, triple.object.boundaries)
        except IndexError:
            # This probably mean the entity span across multiple sentences. This is probably due to the paragraph not
            # being split correctly into sentences. This can happen for example in sentences which contains dot which
            # do not mark end-of-sentence, like "Cedric Kushner Promotions, Ltd. v. King, 533 U.S. 158 (2001)..."
            continue

        if subject_sent_idx == object_sent_index:
            # TODO: In the future, we can just combine the sentences for the specific triplet that contains both.
            filtered_list.append(triple)

    return filtered_list


def _group_triples_by_subject_by_sentence(triples_by_subject: Dict[str, List[Triple]],
                                          sentence_boundaries: List[Tuple[int, int]]
                                          ) -> Dict[int, Dict[str, List[Triple]]]:
    res_dict = {_find_sentence_id_by_token_boundaries(sentence_boundaries, ast.literal_eval(k)): {} for
                k in triples_by_subject}

    for k, triple_list in triples_by_subject.items():
        subject_sent_id = _find_sentence_id_by_token_boundaries(sentence_boundaries, ast.literal_eval(k))
        for triple in triple_list:
            object_sent_id = _find_sentence_id_by_token_boundaries(sentence_boundaries, triple.object.boundaries)
            # assert triple.sentence_id == subject_sent_id
            assert subject_sent_id == object_sent_id

        res_dict[subject_sent_id].update({k: triple_list})

    return res_dict


def _get_words_boundaries_from_surfaceform_boundaries(words_boundaries: List[Tuple[int, int]],
                                                      surfaceform_boundaries: Tuple[int, int]) -> Tuple[int, int]:
    form_start, form_end = surfaceform_boundaries
    form_range = set(range(form_start, form_end))
    words_list = []
    for i, (word_start, word_end) in enumerate(words_boundaries):
        word_range = set(range(word_start, word_end + 1))
        if len(form_range.intersection(word_range)) > 0:
            words_list.append(i)

    assert len(words_list) > 0

    # assert the words in the list are continuous
    span_start, span_end = min(words_list), max(words_list) + 1
    assert set(range(span_start, span_end)) == set(words_list)

    return span_start, span_end


# Also convert the TripletsBySubjectBySentence dictionary, from sentence id key to tokenized sentence string key
def _convert_surfaceform_boundaries_to_word_boundaries(
        triples_by_subject_by_sentence: Dict[int, Dict[str, List[Triple]]],
        sentences_boundaries: List[Tuple[int, int]],
        words_boundaries: List[Tuple[int, int]],
        sentences: List[str]) -> TripletsBySubjectBySentence:
    res_dict: Dict[str, Dict[str, List[Triple]]] = {}
    for sentence_idx in triples_by_subject_by_sentence.keys():
        sent_start, sent_end = sentences_boundaries[sentence_idx]
        sentence_words_boundaries = [(start, end) for (start, end) in words_boundaries
                                     if sent_start <= start < end <= sent_end]
        sentence = sentences[sentence_idx]
        tokenized_sentence = " ".join(
            [sentence[start - sent_start: end - sent_start] for (start, end) in sentence_words_boundaries])

        sentence_dict = {tokenized_sentence: {}}

        triples_by_subject = triples_by_subject_by_sentence[sentence_idx]
        for subject_boundaries_str in triples_by_subject.keys():
            subject_word_boundaries = _get_words_boundaries_from_surfaceform_boundaries(
                sentence_words_boundaries,
                ast.literal_eval(subject_boundaries_str))

            triples_by_subject_dict = {str(subject_word_boundaries): []}

            for triple in triples_by_subject[subject_boundaries_str]:
                triple_copy = copy.deepcopy(triple)
                triple_copy.subject.boundaries = subject_word_boundaries
                triple_copy.object.boundaries = _get_words_boundaries_from_surfaceform_boundaries(
                    sentence_words_boundaries,
                    triple.object.boundaries)

                triples_by_subject_dict[str(subject_word_boundaries)].append(triple_copy)

            sentence_dict[tokenized_sentence].update(triples_by_subject_dict)

        res_dict.update(sentence_dict)

    return res_dict


def parse_rebel_data_file(input_path: str, lang: str) -> Iterator[TripletsBySubjectBySentence]:
    if lang == "thai":
        import pythainlp
        nlp = pythainlp.tokenize
    else:
        nlp = trankit.Pipeline(lang=lang, gpu=True, cache_dir='./cache')
    with open(input_path, 'r', encoding="utf-8") as f:
        for id_, row in tqdm.tqdm(enumerate(f), "reading"):
            article = json.loads(row)
            parsed_doc = Document.fromJSON(article, nlp, lang)

            filtered_triples = _filter_cross_sentence_triples(parsed_doc.triples, parsed_doc.sentences_boundaries)
            if len(filtered_triples) == 0:
                continue
            triples_by_subject = _group_triples_by_subject(filtered_triples)
            triples_by_subject_by_sentence = _group_triples_by_subject_by_sentence(triples_by_subject,
                                                                                   parsed_doc.sentences_boundaries)

            triples_by_subject_by_sentence_word_boundaries = \
                _convert_surfaceform_boundaries_to_word_boundaries(
                    triples_by_subject_by_sentence,
                    parsed_doc.sentences_boundaries,
                    parsed_doc.words_boundaries,
                    parsed_doc.get_sentences())

            yield triples_by_subject_by_sentence_word_boundaries


# endregion


# region Utils
def filter_triples_by_subject_by_sentence_by_relation_uri(
        triples_by_subject_by_sentence: TripletsBySubjectBySentence,
        relation_list: List[str]) -> TripletsBySubjectBySentence:
    res_dict: Dict[str, Dict[str, List[Triple]]] = {}
    for sentence, triples_by_subject in triples_by_subject_by_sentence.items():
        filtered_triples_by_subject_dict = {}
        for subject_boundaries, triple_list in triples_by_subject.items():
            filtered_triple_list = []
            for triple in triple_list:
                if triple.predicate.uri in relation_list:
                    filtered_triple_list.append(triple)
            if len(filtered_triple_list) > 0:
                filtered_triples_by_subject_dict[subject_boundaries] = filtered_triple_list
        if len(filtered_triples_by_subject_dict) > 0:
            res_dict[sentence] = filtered_triples_by_subject_dict

    return res_dict


def triples_by_subject_by_sentence_to_json(triples_by_subject_by_sentence: TripletsBySubjectBySentence) -> str:
    dict_copy = copy.deepcopy(triples_by_subject_by_sentence)

    for sentence, by_subject_dict in dict_copy.items():
        for subject, triple_list in by_subject_dict.items():
            json_triple_list = []
            for triple in triple_list:
                triple_json = triple.toJSON()
                json_triple_list.append(triple_json)
            by_subject_dict[subject] = json_triple_list

    return json.dumps(dict_copy)


def json_to_triples_by_subject_by_sentence(json_line: str) -> TripletsBySubjectBySentence:
    json_dict = json.loads(json_line)

    for sentence, by_subject_dict in json_dict.items():
        for subject, triple_str_list in by_subject_dict.items():
            triple_list = []
            for triple_str in triple_str_list:
                triple_obj = Triple.fromJSON(json.loads(triple_str))
                triple_list.append(triple_obj)
            by_subject_dict[subject] = triple_list

    return json_dict


def triples_by_subject_to_sequence(triples_by_subject: Dict[str, List[Triple]],
                                   use_pointer_format: bool,
                                   use_relation_uri: bool) -> str:
    # TODO: Might want in the future to have a pointer per token and not per word (might be easier for a parser).
    #  On the other hand, it seem the model can handle this format.
    def _get_entity_ptr_str(entity: Entity) -> str:
        entity_start, entity_end = entity.boundaries
        ent_ptr_str = ""
        for i in range(entity_start, entity_end):
            ent_ptr_str = f'{ent_ptr_str} @ptr{i}'
        ent_ptr_str = ent_ptr_str[1:]
        return ent_ptr_str

    decoder_output = '<triplet> '
    # Sort the triples by subject index in the sentence
    triples_by_subject = sorted(triples_by_subject.items(), key=lambda x: ast.literal_eval(x[0])[0])
    for subject_boundaries, triple_list in triples_by_subject:
        subject_obj = triple_list[0].subject
        if use_pointer_format:
            subj_str = _get_entity_ptr_str(subject_obj)
        else:
            subj_str = subject_obj.surfaceform
        decoder_output += subj_str + ' <subj> '
        for triplet in triple_list:
            predicate = triplet.predicate.uri if use_relation_uri else triplet.predicate.surfaceform
            if len(predicate.split()) > 1:
                predicate = "_".join(predicate.split())
            if use_pointer_format:
                obj_str = _get_entity_ptr_str(triplet.object)
            else:
                obj_str = triplet.object.surfaceform
            decoder_output += obj_str + ' <obj> ' + predicate + ' <subj> '
        decoder_output = decoder_output[:-len(' <subj> ')]
        decoder_output += ' <triplet> '
    decoder_output = decoder_output[:-len(' <triplet> ')]

    return decoder_output


@dataclass
class Relation:
    subject: str
    object: str
    predicate: str


def parse_sequence_into_relations(sequence: str) -> List[Relation]:
    relation_list: List[Relation] = list()

    if not sequence.startswith("<triplet>"):
        return list()

    triplets_list = sequence.split("<triplet>")
    for triplet_str in triplets_list:
        if '<subj>' not in sequence or sequence.startswith('<subj>'):
            return list()

        split = triplet_str.split('<subj>')
        subject = split[0]
        obj_rel_str_list = split[1:]
        for obj_rel_str in obj_rel_str_list:
            split = obj_rel_str.split('<obj>')

            if len(split) != 2:
                return list()
            obj, predicate = split

            rel = Relation(subject, obj, predicate)
            relation_list.append(rel)

    return relation_list


def split_dataset_to_train_dev_test(triples_by_subject_by_sentence_list: List[TripletsBySubjectBySentence],
                                    train_proportion: float,
                                    dev_proportion: float
                                    ) -> Dict[str, List[TripletsBySubjectBySentence]]:
    # We split by sentence. We need to split by relation as well, but for now we will make the assumption
    # random split is good enough
    assert train_proportion + dev_proportion < 1
    dataset_len = len(triples_by_subject_by_sentence_list)
    train_size = int(dataset_len * train_proportion)
    dev_size = int(dataset_len * dev_proportion)

    train_split = []
    dev_split = []
    test_split = []

    temp = triples_by_subject_by_sentence_list
    random.shuffle(temp)

    for i, instance in enumerate(temp):
        if i < train_size:
            train_split.append(instance)
        elif i < train_size + dev_size:
            dev_split.append(instance)
        else:
            test_split.append(instance)

    return {"train": train_split, "dev": dev_split, "test": test_split}


def create_vocab_from_seq2seq_file(file_path_list: List[str], output_dir: str) -> None:
    ontology_tokens = ['@@UNKNOWN@@', '@@PADDING@@', '@start@', '@end@', '<triplet>', '<subj>', '<obj>']
    pointers_tokens = set()
    rel_tokens = set()

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
                    if token.startswith('<'):
                        assert token in ontology_tokens
                    elif token.startswith('@ptr'):
                        pointers_tokens.add(token)
                    else:
                        rel_tokens.add(token)

    with open(f'{output_dir}/target_tokens.txt', 'w', encoding='utf-8') as f:
        for token in ontology_tokens:
            f.write(f'{token}\n')
        for token in sorted(list(rel_tokens)):
            f.write(f'{token}\n')

        pointer_tokens_size = round((len(pointers_tokens) + 100) / 100) * 100
        for i in range(pointer_tokens_size):
            f.write(f'@ptr{i}\n')

    with open(f'{output_dir}/non_padded_namespaces.txt', 'w', encoding='utf-8') as f:
        for token in ["*tags", "*labels"]:
            f.write(f'{token}\n')


# endregion


# region Reorder Funcs

def reorder_triples_by_subject_by_sentence_object_w_word_boundaries(
        triples_by_subject_by_sentence: TripletsBySubjectBySentence,
        reorder_algo: UdReorderingAlgo,
        reorder_by_lang: str) -> Dict[str, Dict[str, List[Triple]]]:
    # TODO: In the future, generalize the Reordering Algo to an interface for the other reordering algorithm
    reordered_triples_by_subject_by_sentence_word_boundaries = {}

    for sentence, triples_by_subject in triples_by_subject_by_sentence.items():
        entities_list = []
        for subject_boundaries, triple_list in triples_by_subject.items():
            for triple in triple_list:
                subject_dict = {
                    'surfaceform': triple.subject.surfaceform,
                    'inclusive_span': (triple.subject.boundaries[0], triple.subject.boundaries[1] - 1)
                }
                object_dict = {
                    'surfaceform': triple.object.surfaceform,
                    'inclusive_span': (triple.object.boundaries[0], triple.object.boundaries[1] - 1)
                }
                entities_list.extend([subject_dict, object_dict])

        mapping = reorder_algo.get_entities_aware_reorder_mapping(sentence, reorder_by_lang, entities_list)
        if mapping is not None:
            reordered_sentence = reorder_algo.reorder_sentence(sentence, mapping)
            reordered_triples_by_subject = {}
            for subject_boundaries, triple_list in triples_by_subject.items():
                reordered_triple_list = copy.deepcopy(triple_list)
                reordered_subject_boundaries = ""
                for triple in reordered_triple_list:
                    subj_start, subj_end_inclusive = triple.subject.boundaries[0], triple.subject.boundaries[
                        1] - 1
                    reordered_inclusive_span = reorder_algo.get_continuous_mapped_span((subj_start,
                                                                                        subj_end_inclusive),
                                                                                       mapping)
                    triple.subject.boundaries = (reordered_inclusive_span[0], reordered_inclusive_span[1] + 1)

                    if reordered_subject_boundaries == "":
                        reordered_subject_boundaries = triple.subject.boundaries
                    else:
                        assert reordered_subject_boundaries == triple.subject.boundaries

                    obj_start, obj_end_inclusive = triple.object.boundaries[0], triple.object.boundaries[1] - 1
                    reordered_inclusive_span = reorder_algo.get_continuous_mapped_span((obj_start,
                                                                                        obj_end_inclusive),
                                                                                       mapping)

                    triple.object.boundaries = (reordered_inclusive_span[0], reordered_inclusive_span[1] + 1)

                reordered_triples_by_subject[str(reordered_subject_boundaries)] = reordered_triple_list
            reordered_triples_by_subject_by_sentence_word_boundaries[
                reordered_sentence] = reordered_triples_by_subject

    return reordered_triples_by_subject_by_sentence_word_boundaries


# endregion


# region Dataset Creation Utils

def get_dataset_stats_from_file(json_dataset_path: str, label: str = "") -> pd.DataFrame:
    rebel_instances = []
    with open(json_dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f.readlines(), "reading json"):
            json_obj = json_to_triples_by_subject_by_sentence(line)
            rebel_instances.append(json_obj)

    return get_dataset_stats(rebel_instances, label)


def get_dataset_stats(rebel_instances: List[TripletsBySubjectBySentence], label: str = "") -> pd.DataFrame:
    sentence_count = 0
    subject_per_sentence_counter = Counter()
    triple_per_sentence_count = Counter()
    rel_counter = Counter()
    for triples_by_subject_by_sentence in rebel_instances:
        for sentence, triples_by_subject in triples_by_subject_by_sentence.items():
            sentence_count += 1
            subject_per_sentence_counter[len(triples_by_subject)] += 1
            triple_count = 0

            for subject, triple_list in triples_by_subject.items():
                triple_count += len(triple_list)
                for triple in triple_list:
                    rel = triple.predicate.uri
                    rel_counter[rel] += 1
            triple_per_sentence_count[triple_count] += 1

    import pandas as pd
    rel_list = list(rel_counter.keys())
    df = pd.DataFrame(index=rel_list)
    df[f'counter_{label}'] = df.index.map(lambda val: rel_counter[val])
    return df


def creat_json_dataset_subset(json_dataset_path: str, output_dir: str, size: int, rel_filter: List[str]):
    top_k_rel = rel_filter

    rebel_instances = []
    with open(json_dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f.readlines(), "reading json"):
            json_obj = json_to_triples_by_subject_by_sentence(line)
            rebel_instances.append(json_obj)

    rel_counter = Counter()
    per_rel_limit = size / len(top_k_rel)
    selected_instances = []
    unused_instances = []
    for triples_by_subject_by_sentence in rebel_instances:
        for sentence, triples_by_subject in triples_by_subject_by_sentence.items():
            instance_selected = False
            relevant_instance = False
            for subject, triple_list in triples_by_subject.items():
                for triple in triple_list:
                    rel = triple.predicate.uri

                    if rel in top_k_rel:
                        relevant_instance = True
                    if rel in top_k_rel and \
                            (rel_counter[rel] <= per_rel_limit
                             or all([rel_counter[rel] > per_rel_limit for rel in top_k_rel])):
                        instance_selected = True

                if instance_selected:
                    for triple in triple_list:
                        rel = triple.predicate.uri
                        rel_counter[rel] += 1

            if instance_selected:
                selected_instances.append({sentence: copy.deepcopy(triples_by_subject)})
            elif relevant_instance:
                unused_instances.append({sentence: copy.deepcopy(triples_by_subject)})

        if len(selected_instances) > size:
            break

    if len(selected_instances) < size:
        to_fill_count = size - len(selected_instances)
        random.shuffle(unused_instances)
        selected_instances.extend(unused_instances[:to_fill_count])

    for triples_by_subject_by_sentence in selected_instances:
        for sentence, triples_by_subject in triples_by_subject_by_sentence.items():
            filtered_triples_by_subject_list = {}
            for subject, triple_list in triples_by_subject.items():
                filtered_triple_list: List[Triple] = []
                for triple in triple_list:
                    rel = triple.predicate.uri
                    if rel in top_k_rel:
                        filtered_triple_list.append(triple)
                if len(filtered_triple_list) > 0:
                    filtered_triples_by_subject_list[subject] = filtered_triple_list

            triples_by_subject_by_sentence[sentence] = filtered_triples_by_subject_list

    basename, ext = os.path.basename(json_dataset_path).split(".")

    actual_size = len(selected_instances)
    with open(f'{output_dir}/{basename}_{actual_size}.{ext}', 'x', encoding='utf-8') as f:
        for instance in tqdm.tqdm(selected_instances, "writing json"):
            json_str = triples_by_subject_by_sentence_to_json(instance)
            f.write(f'{json_str}\n')


def get_top_k_rel_from_json_file(json_dataset_path: str, k: int):
    stats_df = get_dataset_stats_from_file(json_dataset_path)
    top_k_rel = stats_df.sort_values(by="counter_", ascending=False).index[0:k]

    return top_k_rel


def get_rel_with_min_occurrence_json_file(json_dataset_path: str, k: int):
    stats_df = get_dataset_stats_from_file(json_dataset_path)
    top_k_rel = stats_df[stats_df["counter_"] >= k].index

    return top_k_rel


# TODO: Fix reordering! Go over the exception and solve it. For example entities with "-". Probably just remove it and replace it
#  with " ". The span will be the same and my code will add the "-' again. Need to remember where it was thou.
def create_json_rebel_dataset(input_lang: str, input_fie: str, output_dir: str,
                              split: bool, train_proportion: float = 0.8, dev_proportion: float = 0.1):
    rebel_instances = list(parse_rebel_data_file(input_fie, input_lang))

    if split:
        instance_splits = split_dataset_to_train_dev_test(rebel_instances, train_proportion, dev_proportion)
    else:
        instance_splits = {"test": rebel_instances}

    for split, instances in instance_splits.items():
        output_file_path = f'{output_dir}/{input_lang}_{split}.jsonl'
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for triples_by_subject_by_sentence in tqdm.tqdm(instances, "writing json"):
                json_str = triples_by_subject_by_sentence_to_json(triples_by_subject_by_sentence)
                f.write(f'{json_str}\n')


def create_seq2seq_rebel_dataset(input_json_path: str,
                                 output_dir: str,
                                 input_lang: Optional[str] = None,
                                 reorder_algo: Optional[UdReorderingAlgo.ReorderAlgo] = None,
                                 reorder_by_lang: Optional[str] = None,
                                 use_pointers: bool = True,
                                 use_uri_rel: bool = True):
    rebel_instances = []
    with open(input_json_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f.readlines(), "reading json"):
            json_obj = json_to_triples_by_subject_by_sentence(line)
            rebel_instances.append(json_obj)

    # We can't know in advance how many sentences there are in a single instance, so we take the lower bound
    # of a single sentence per instance. We filter the list this early, so to save on the expensive reordering
    # operation.

    if reorder_algo:
        assert input_lang
        reorder_alog = UdReorderingAlgo(reorder_algo, input_lang)
    else:
        reorder_algo = None

    seq2seq_std_strings = []
    seq2seq_reordered_strings = []

    error_dict = defaultdict(int)
    for i, triples_by_subject_by_sentence in tqdm.tqdm(enumerate(rebel_instances), "writing tsv"):
        if reorder_by_lang is not None:
            try:
                reordered_triples_by_subject_by_sentence = \
                    reorder_triples_by_subject_by_sentence_object_w_word_boundaries(
                        triples_by_subject_by_sentence, reorder_alog, reorder_by_lang)
            except Exception as e:
                raise e
                error_dict[f'error reordering instance: {str(e)}'] += 1
                reordered_triples_by_subject_by_sentence = triples_by_subject_by_sentence

            for sentence, triples_by_subject in reordered_triples_by_subject_by_sentence.items():
                target_seq = triples_by_subject_to_sequence(triples_by_subject, use_pointers, use_uri_rel)
                sentence = sentence.strip("\n")
                sentence = sentence.replace("\n", " ")
                assert '\n' not in sentence
                assert len(sentence) > 0 and len(target_seq) > 0
                seq2seq_reordered_strings.append(f'{sentence}\t{target_seq}\n')

        for sentence, triples_by_subject in triples_by_subject_by_sentence.items():
            target_seq = triples_by_subject_to_sequence(triples_by_subject, use_pointers, use_uri_rel)
            sentence = sentence.strip("\n")
            sentence = sentence.replace("\n", " ")
            assert '\n' not in sentence
            assert len(sentence) > 0 and len(target_seq) > 0
            seq2seq_std_strings.append(f'{sentence}\t{target_seq}\n')

    print(error_dict)

    filename = os.path.basename(input_json_path).split(".")[0]

    output_file_path = f'{output_dir}/{filename}'

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



# endregion

# region Dataset Creation Scrips

def create_json_datasets_script():
    rebel_data_dict = {
        "english": "experiments/datasets/relation_extraction/rebel/en.jsonl",
        "hindi": "experiments/datasets/relation_extraction/rebel/hi.jsonl",
        "korean": "experiments/datasets/relation_extraction/rebel/ko.jsonl",
        "thai": "experiments/datasets/relation_extraction/rebel/th.jsonl",
        "vietnamese": "experiments/datasets/relation_extraction/rebel/vi.jsonl",
    }
    output_dir = "experiments/processed_datasets/rebel/json_format/"
    os.makedirs(output_dir, exist_ok=True)

    for lang, path in rebel_data_dict.items():
        create_json_rebel_dataset(lang, path, output_dir, split=(lang == "english"))


def create_json_datasets_small_script():
    input_dir = "experiments/processed_datasets/rebel/json_format/"
    output_dir = "experiments/processed_datasets/rebel/json_format_small/"
    os.makedirs(output_dir, exist_ok=True)

    english_train_json_file = "experiments/processed_datasets/rebel/json_format/english_train.jsonl"

    # > 250 instance per relationship, means top 118 rel
    top_rel = get_rel_with_min_occurrence_json_file(english_train_json_file, 250)
    print(f'rel count: {len(top_rel)}')

    for file_path in glob.glob(f'{input_dir}/*.jsonl'):
        basename = os.path.basename(file_path)
        # dataset_size = sys.maxsize if "train" in basename else 2000 if "dev" in basename else 10000
        dataset_size = 30000 if "train" in basename else 2000 if "dev" in basename else 10000
        if "train" not in basename:
            continue
        creat_json_dataset_subset(file_path, output_dir, dataset_size, top_rel)


def create_standard_seq2seq_datasets_script():
    input_dir = "experiments/processed_datasets/rebel/json_format_small/"
    output_dir = "experiments/processed_datasets/rebel/seq2seq_standard/"
    os.makedirs(output_dir, exist_ok=True)

    for file_path in glob.glob(f'{input_dir}/*.jsonl'):
        if "50000" not in file_path:
            continue
        create_seq2seq_rebel_dataset(file_path, output_dir)


def create_reordered_seq2seq_datasets_script():
    file_path_regex = f'experiments/processed_datasets/rebel/json_format_small/english_train_50000.jsonl'
    for file_path in glob.glob(file_path_regex):
        for lang in ["hindi", "korean", "vietnamese"]:
            output_dir = f'experiments/processed_datasets/rebel/seq2seq_english_reordered_by_{lang}/'
            os.makedirs(output_dir, exist_ok=True)

            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI, UdReorderingAlgo.ReorderAlgo.RASOOLINI]:
                print(
                    f'Creating seq2seq dataset. lang: {lang}, file: {os.path.basename(file_path)}, algorithm: {reorder_algo.name}')
                create_seq2seq_rebel_dataset(file_path,
                                             output_dir,
                                             "english",
                                             reorder_algo=reorder_algo,
                                             reorder_by_lang=lang)

def create_reordered_seq2seq_datasets_script_gurobi_temp():
    file_path_regex = f'experiments/processed_datasets/rebel/json_format_small/english_train_144976.jsonl'
    for file_path in glob.glob(file_path_regex):
        for lang in ["hindi", "vietnamese", "korean"]:
            output_dir = f'experiments/processed_datasets/rebel/seq2seq_english_reordered_by_{lang}/'
            os.makedirs(output_dir, exist_ok=True)

            for reorder_algo in [UdReorderingAlgo.ReorderAlgo.HUJI_GUROBI]:
                print(
                    f'Creating seq2seq dataset. lang: {lang}, file: {os.path.basename(file_path)}, algorithm: {reorder_algo.name}')
                create_seq2seq_rebel_dataset(file_path,
                                             output_dir,
                                             "english",
                                             reorder_algo=reorder_algo,
                                             reorder_by_lang=lang)


# endregion

if __name__ == "__main__":
    # Example Usage
    print("Creating datasets:")
    # create_json_datasets_script()
    # create_json_datasets_small_script()
    # create_standard_seq2seq_datasets_script()
    # create_reordered_seq2seq_datasets_script()

    # create_vocab_from_seq2seq_file(["experiments/processed_datasets/rebel/seq2seq_standard/english_train_144976.tsv",
    #                                 "experiments/processed_datasets/rebel/seq2seq_standard/english_dev_2001.tsv",
    #                                 "experiments/processed_datasets/rebel/seq2seq_standard/english_test_10000.tsv"],
    #                                "experiments/vocabs/rebel_pointers")

    create_reordered_seq2seq_datasets_script_gurobi_temp()
