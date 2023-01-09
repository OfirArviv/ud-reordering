from typing import Optional, List

import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("attachment_scores_custom")
class AttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """
    def __init__(self, ignore_classes: List[int] = None,
                 ignore_directions: List[str] = None) -> None:
        self._labeled_correct = 0.
        self._unlabeled_correct = 0.
        self._label_only_correct = 0.
        self._exact_labeled_correct = 0.
        self._exact_unlabeled_correct = 0.
        self._exact_label_only_correct = 0.
        self._total_words = 0.
        self._total_sentences = 0.

        self._ignore_classes: List[int] = ignore_classes or []
        self._ignore_directions: List[int] = ignore_directions or []

    def __call__(self, # type: ignore
                 predicted_indices: torch.Tensor,
                 predicted_labels: torch.Tensor,
                 gold_indices: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predicted_indices``.
        """
        detached = self.detach_tensors(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = detached

        mask = mask.long()
        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        """
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask * (~label_mask).long()
        """

        for label_to_mask, direction_to_mask in zip(self._ignore_classes, self._ignore_directions):
            label_mask = gold_labels.eq(label_to_mask)

            if direction_to_mask == "forward":
                direction_mask = gold_indices < torch.Tensor(list(range(1, gold_indices.shape[1]+1)))
            elif direction_to_mask == "backward":
                direction_mask = gold_indices > torch.Tensor(list(range(1, gold_indices.shape[1]+1)))
            elif direction_to_mask == "all":
                direction_mask = torch.ones_like(gold_indices, dtype=torch.bool)
            else:
                assert False

            mask = mask*(~(label_mask*direction_mask)).long()

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + (1 - mask)).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + (1 - mask)).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum()
        self._labeled_correct += correct_labels_and_indices.sum()
        self._exact_labeled_correct += labeled_exact_match.sum()
        self._label_only_correct += correct_labels.sum()
        self._exact_label_only_correct += (correct_labels + (1 - mask)).prod(dim=-1).sum()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (1 - mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0
        label_only_attachment_score = 0.0
        label_only_exact_match = 0.0

        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
            label_only_attachment_score = float(self._label_only_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(self._total_sentences)
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
            label_only_exact_match = float(self._exact_label_only_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        return {
                "UAS": unlabeled_attachment_score,
                "LAS": labeled_attachment_score,
                "UEM": unlabeled_exact_match,
                "LEM": labeled_exact_match,
                "LOAS": label_only_attachment_score,
                "LOEM": label_only_exact_match
        }

    def reset(self):
        self._labeled_correct = 0.
        self._unlabeled_correct = 0.
        self._exact_labeled_correct = 0.
        self._exact_unlabeled_correct = 0.
        self._total_words = 0.
        self._total_sentences = 0.
        self._label_only_correct = 0.
        self._exact_label_only_correct = 0.