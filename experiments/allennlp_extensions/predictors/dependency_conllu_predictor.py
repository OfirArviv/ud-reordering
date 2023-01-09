from collections import OrderedDict
from typing import List, Optional, Dict
import torch
from conllu import TokenList
from overrides import overrides
from allennlp.training.metrics.metric import Metric


def find_by_id(id: int, token_list: TokenList, word: str = None):
    for token in token_list:
        if isinstance(token["id"], int) and int(token["id"]) == id:
            if word is not None:
                assert token["form"] == word
            return token
    raise KeyError(f'Can not find id {id} in token list with metadata {token_list.metadata}.')


def create_token(id: str, form: str, head: int, deprel: str,
                lemma: str = None, upostag: str = None, xpostag: str = None, ):
    d = OrderedDict()
    d["id"] = id
    d["form"] = form
    d["lemma"] = lemma
    d["upostag"] = upostag
    d["xpostag"] = xpostag
    d["feats"] = None
    d["head"] = head
    d["deprel"] = deprel
    d["deps"] = None
    d["misc"] = None

    return d


@Metric.register("dependency_predictor_conllu")
class DependencyConlluPredictor(Metric):
    def __init__(self, index_to_label_dict: Dict[int, str],
                 output_path: str = None) -> None:
        self.index_to_label_dict = index_to_label_dict
        self.output_path = output_path

        self.predictions = []

    @overrides
    def __call__(
            self,
            words: List[str],
            pos: List[str],
            ids: List[int],
            annotation: TokenList,
            predicted_indices: torch.Tensor,
            predicted_labels: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            confidence: Optional[torch.Tensor] = None
    ):
        sentence_length = mask.sum()
        predicted_indices = predicted_indices.numpy()[:sentence_length]
        predicted_labels = predicted_labels.numpy()[:sentence_length]
        predicted_labels = list(map(lambda x: self.index_to_label_dict[x], predicted_labels))
        if confidence is not None:
            confidence = confidence.numpy()[:sentence_length]
        else:
            confidence = [0 for c in predicted_indices]

        for word, pos, id, head, label, conf in zip(words, pos, ids, predicted_indices, predicted_labels, confidence):
            token = find_by_id(id, annotation, word)
            token["head"] = head
            token["deprel"] = label
            token["misc"] = f'confidence={conf}'

        self.predictions.append(annotation)

    def get_metric(self, reset: bool = False):

        if reset:
            with open(self.output_path, 'w') as file:
                for token_list in self.predictions:
                    file.write(token_list.serialize())

            self.reset()

        return {"predictions": len(self.predictions)}

    def reset(self) -> None:
        self.predictions = []
