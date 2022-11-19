import copy
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from overrides import overrides
import torch
from allennlp.training.metrics.metric import Metric


@dataclass
class Relation:
    subject: str
    object: str
    predicate: str


@Metric.register("re_fscore")
class RelationExtractionFScores(Metric):
    """
    Return the micro and macro f-scores for a relation extraction sequence in the format described
    in the paper: "REBEL: Relation Extraction By End-to-end Language generation".

    Code taken from: https://github.com/btaille/sincere/blob/master/code/utils/evaluation.py
    """

    def __init__(self) -> None:
        # Notice: Because the relation_types list updates are more scores are put in. The macro-f-score is not final
        # until going over all the examples.
        self._relation_types: List[str] = []
        self._scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in self._relation_types + ["ALL"]}

    def __call__(
            self,
            predictions: List[List[str]],
            gold_labels: List[List[str]],
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `str`, required.
            A string array of predictions of shape (batch_size, ...).
        gold_labels : `str`, required.
            A string array of the same shape as `predictions`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predictions`. This is not used and is here for
            compatibility with AllenNlp 'Metric' api.
        """

        # Some sanity checks.
        if len(gold_labels) != len(predictions):
            raise ValueError(
                f"gold_labels must have shape == len(predictions) but "
                f"found tensor of shape: {len(gold_labels)}"
            )

        gt_relations = [self.parse_sequence_into_relations(" ".join(gold_label)) for gold_label in gold_labels]
        pred_relations = [self.parse_sequence_into_relations(" ".join(prediction)) for prediction in predictions]

        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(pred_relations, gt_relations):
            for rel in pred_sent + gt_sent:
                if rel.predicate not in self._relation_types:
                    self._relation_types.append(rel.predicate)
                    self._scores.update({rel.predicate: {"tp": 0, "fp": 0, "fn": 0}})

            for rel_type in self._relation_types:
                pred_rels = {(rel.subject, rel.object) for rel in pred_sent if rel.predicate == rel_type}
                gt_rels = {(rel.subject, rel.object) for rel in gt_sent if rel.predicate == rel_type}

                self._scores[rel_type]["tp"] += len(pred_rels & gt_rels)
                self._scores[rel_type]["fp"] += len(pred_rels - gt_rels)
                self._scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated metrics for all the relations.
        """

        # Compute per entity Precision / Recall / F1
        for rel_type in self._scores.keys():
            if self._scores[rel_type]["tp"]:
                self._scores[rel_type]["p"] = 100 * self._scores[rel_type]["tp"] / (
                        self._scores[rel_type]["fp"] + self._scores[rel_type]["tp"])
                self._scores[rel_type]["r"] = 100 * self._scores[rel_type]["tp"] / (
                        self._scores[rel_type]["fn"] + self._scores[rel_type]["tp"])
            else:
                self._scores[rel_type]["p"], self._scores[rel_type]["r"] = 0, 0

            if not self._scores[rel_type]["p"] + self._scores[rel_type]["r"] == 0:
                self._scores[rel_type]["f1"] = 2 * self._scores[rel_type]["p"] * self._scores[rel_type]["r"] / (
                        self._scores[rel_type]["p"] + self._scores[rel_type]["r"])
            else:
                self._scores[rel_type]["f1"] = 0

        # Compute micro F1 Scores
        tp = sum([self._scores[rel_type]["tp"] for rel_type in self._relation_types])
        fp = sum([self._scores[rel_type]["fp"] for rel_type in self._relation_types])
        fn = sum([self._scores[rel_type]["fn"] for rel_type in self._relation_types])

        if tp:
            precision = 100 * tp / (tp + fp)
            recall = 100 * tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        else:
            precision, recall, f1 = 0, 0, 0

        self._scores["ALL"]["p"] = precision
        self._scores["ALL"]["r"] = recall
        self._scores["ALL"]["f1"] = f1
        self._scores["ALL"]["tp"] = tp
        self._scores["ALL"]["fp"] = fp
        self._scores["ALL"]["fn"] = fn

        # Compute Macro F1 Scores
        self._scores["ALL"]["Macro_f1"] = np.mean([self._scores[ent_type]["f1"] for ent_type in self._relation_types])
        self._scores["ALL"]["Macro_p"] = np.mean([self._scores[ent_type]["p"] for ent_type in self._relation_types])
        self._scores["ALL"]["Macro_r"] = np.mean([self._scores[ent_type]["r"] for ent_type in self._relation_types])

        output = copy.deepcopy(self._scores['ALL'])

        if reset:
            self.reset()

        return output

    @overrides
    def reset(self):
        self._relation_types = []
        self._scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in self._relation_types + ["ALL"]}

    @staticmethod
    def parse_sequence_into_relations(sequence: str) -> List[Relation]:
        relation_list: List[Relation] = list()

        if not sequence.startswith("<triplet>"):
            return list()

        triplets_list = sequence.split("<triplet>")
        assert triplets_list[0] == ""
        triplets_list = triplets_list[1:]
        for triplet_str in triplets_list:
            triplet_str = triplet_str.strip()
            if '<subj>' not in triplet_str or triplet_str.startswith('<subj>'):
                return list()

            split = triplet_str.split('<subj>')
            subject = split[0].strip()
            obj_rel_str_list = split[1:]
            for obj_rel_str in obj_rel_str_list:
                obj_rel_str.strip()
                split = obj_rel_str.split('<obj>')

                if len(split) != 2:
                    return list()
                obj, predicate = split
                obj = obj.strip()
                predicate = predicate.strip()

                rel = Relation(subject, obj, predicate)
                relation_list.append(rel)

        return relation_list
