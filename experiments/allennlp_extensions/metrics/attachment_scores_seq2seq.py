from typing import Optional
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("attachment_scores_seq2seq")
class Seq2SeqAttachmentScores(Metric):

    def __init__(self) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

    def __call__(
            self,
            predictions,
            gold_labels,
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predictions`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predictions`.
        """

        # Some sanity checks.
        if len(gold_labels) != len(predictions):
            raise ValueError(
                f"gold_labels must have shape == len(predictions) but "
                f"found tensor of shape: {len(gold_labels)}"
            )

        for p, g in zip(predictions, gold_labels):
            instance_correct_unlabeled_count = 0
            instance_correct_labeled_count = 0

            p_arr = p.split()
            g_arr = g.split()

            assert len(p_arr) == len(g_arr)

            for i in range(start=0, stop=len(p_arr), step=2):
                p_position = p_arr[i]
                p_label = p_arr[i+1]

                g_position = g_arr[i]
                g_label = g_arr[i + 1]

                if p_position == g_position:
                    instance_correct_unlabeled_count += 1

                    if p_label == g_label:
                        instance_correct_labeled_count += 1

            sentence_length = len(p_arr) / 2
            self._unlabeled_correct += instance_correct_unlabeled_count
            self._exact_unlabeled_correct += 1 if instance_correct_unlabeled_count == sentence_length else 0
            self._labeled_correct += instance_correct_labeled_count
            self._exact_labeled_correct += 1 if instance_correct_labeled_count == sentence_length else 0
            self._total_sentences += 1
            self._total_words += sentence_length

    def get_metric(
        self,
        reset: bool = False,
    ):
        """
        # Returns
        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0

        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(
                self._total_sentences
            )
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        metrics = {
            "UAS": unlabeled_attachment_score,
            "LAS": labeled_attachment_score,
            "UEM": unlabeled_exact_match,
            "LEM": labeled_exact_match,
        }
        return metrics

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

