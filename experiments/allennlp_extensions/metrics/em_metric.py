from typing import Optional

from overrides import overrides
import torch

from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric


@Metric.register("em_accuracy")
class ExactMatchAccuracy(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.
    That is, if your prediction has shape (batch_size, dim_1, ..., dim_n), this metric considers that
    as a set of `batch_size` predictions and checks that each is *entirely* correct across the remaining dims.
    This means the denominator in the accuracy computation is `batch_size`, with the caveat that predictions
    that are totally masked are ignored (in which case the denominator is the number of predictions that have
    at least one unmasked element).

    This is similar to [`CategoricalAccuracy`](./categorical_accuracy.md), if you've already done a `.max()`
    on your predictions.  If you have categorical output, though, you should typically just use
    `CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    `CategoricalAccuracy`, which assumes a final dimension of size `num_classes`.
    """

    def __init__(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0

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

        # At this point, predictions is (batch_size, rest_of_dims_combined),
        # so .eq -> .prod will be 1 if every element of the instance prediction is correct
        # and 0 if at least one element of the instance prediction is wrong.
        # Because of how we're handling masking, masked positions are automatically "correct".
        correct = 0
        for p, g in zip(predictions, gold_labels):
            if p == g:
                correct = correct + 1

        _correct_count = correct
        _total_count = len(predictions)

        self._correct_count += _correct_count
        self._total_count += _total_count

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated accuracy.
        """
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return {"em_accuracy": accuracy}

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
