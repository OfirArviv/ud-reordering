from typing import Optional, Dict, Any, List, Tuple
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.modules.transformer.t5 import  T5Output, IntT, BoolT

from allennlp.training.metrics import Metric
from overrides import overrides
from transformers import MT5ForConditionalGeneration


@Model.register("mt5")
class MT5(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        token_based_metric: List[Metric] = None,
        beam_size: int = 4,
        max_steps: int = 100,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._model_name = model_name
        # We only instantiate this when we need it.
        self._tokenizer: Optional[PretrainedTransformerTokenizer] = None

        self.t5 = MT5ForConditionalGeneration.from_pretrained(model_name)

        self._beam_size = beam_size
        self._max_steps = max_steps

        self._token_based_metric = token_based_metric

    def _post_load_state_dict(
        self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> Tuple[List[str], List[str]]:
        missing_keys_to_ignore = [
            "t5.encoder.token_embeddings.weight",
            "t5.decoder.token_embeddings.weight",
        ]
        # if self.t5._tie_word_embeddings:
        #     missing_keys_to_ignore.append("t5.lm_head.weight")
        for key in missing_keys_to_ignore:
            if key in missing_keys:
                missing_keys.remove(key)
        return missing_keys, unexpected_keys

    @property
    def tokenizer(self) -> PretrainedTransformerTokenizer:
        if self._tokenizer is None:
            self._tokenizer = PretrainedTransformerTokenizer(self._model_name)
        return self._tokenizer

    def forward(  # type: ignore
        self, source_tokens: TextFieldTensors, target_tokens: Optional[TextFieldTensors] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of T5.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key/namespace.

        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are also stored under the `tokens` key/namespace.
            If no target tokens are given during training / validation, the source tokens are shifted
            to the right by 1.

        # Returns

        `Dict[str, torch.Tensor]`
            Contains the `loss` when `target_tokens` is provided.
            And during prediction, includes `predictions` and `predicted_log_probs` from beam search.

        """
        input_ids, attention_mask = (
            source_tokens["tokens"]["token_ids"],
            source_tokens["tokens"]["mask"],
        )
        labels: Optional[IntT] = None
        decoder_attention_mask: Optional[BoolT] = None
        if target_tokens is not None:
            labels, decoder_attention_mask = (
                target_tokens["tokens"]["token_ids"],  # type: ignore[assignment]
                target_tokens["tokens"]["mask"],  # type: ignore[assignment]
            )
        elif self.training:
            raise ValueError("'target_tokens' required during training")

        output_dict: Dict[str, torch.Tensor] = {}

        if self.training:
            output: T5Output = self.t5(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )
            assert output.loss is not None
            output_dict["loss"] = output.loss
        else:
            output = self.t5.generate(input_ids, num_beams=self._beam_size, max_length=self._max_steps,
                                      output_scores=True, return_dict_in_generate=True)

            output_dict["predictions"] = output['sequences']

            if labels is not None:
                output_dict["loss"] = torch.zeros(1)
                output_dict["labels"] = labels

                if self._token_based_metric is not None:
                    output_dict = self.make_output_human_readable(output_dict)
                    predicted_tokens = output_dict["predicted_text"]

                    for metric in self._token_based_metric:
                        metric(predicted_tokens, output_dict["target_text"])  # type: ignore

        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        predictions = output_dict["predictions"]
        output_dict["predicted_text"] = self.tokenizer.tokenizer.batch_decode(
            predictions, skip_special_tokens=True  # type: ignore[attr-defined]
        )
        labels = output_dict["labels"]
        output_dict["target_text"] = self.tokenizer.tokenizer.batch_decode(
            labels, skip_special_tokens=True  # type: ignore[attr-defined]
        )
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._token_based_metric is not None:
                for metric in self._token_based_metric:
                    all_metrics.update(metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    default_predictor = "seq2seq"
