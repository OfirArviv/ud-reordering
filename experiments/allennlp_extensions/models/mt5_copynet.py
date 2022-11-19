import copy
import warnings
from typing import Optional, Dict, Any, Union, List, Tuple, cast
import torch.nn.functional as F
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.modules.attention import BilinearAttention
from allennlp.nn import util
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.lazy import Lazy
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.modules import Embedding, Seq2SeqEncoder, FeedForward
from allennlp.modules.transformer.t5 import T5Output, KeyValueStates
from allennlp.modules.transformer.util import IntT, BoolT
from allennlp.nn.activations import LinearActivation
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.checkpoint import CheckpointWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import ROUGE, BLEU, BooleanAccuracy
from datasets import Metric
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import add_start_docstrings, T5Config, MT5Config  # ,logger
from transformers.file_utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import T5_START_DOCSTRING, T5PreTrainedModel, PARALLELIZE_DOCSTRING, \
    DEPARALLELIZE_DOCSTRING, T5_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, __HEAD_MASK_WARNING_MSG, T5ForConditionalGeneration, \
    T5Stack
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from experiments.allennlp_extensions.modules.bilinear_attention_gpu import BilinearAttentionCuda


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGenerationWithCopyMechanism(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config,
                 vocab: Vocabulary,
                 target_embedder: Embedding,
                 target_namespace: str = "target_tokens",
                 pointer_vocab_size: int = 200,
                 ):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, target_embedder)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # region CopyNet
        self._vocab = vocab

        self._target_embedder = target_embedder
        self._target_namespace = target_namespace

        if self._target_embedder.get_output_dim() != config.d_model:
            raise ConfigurationError(
                f'Target Embedder output_dim ({self._target_embedder.get_output_dim()}) doesnt match decoder modules input ({config.d_model}).'
            )

        if self.device.type == 'cpu':
            self._copy_attention = BilinearAttention(vector_dim=decoder_config.d_model,
                                                     matrix_dim=encoder_config.d_model,
                                                     normalize=False)
        else:
            self._copy_attention = BilinearAttentionCuda(vector_dim=decoder_config.d_model,
                                                         matrix_dim=encoder_config.d_model,
                                                         normalize=False)

        target_vocab_size = self._vocab.get_vocab_size(target_namespace)
        self._ontology_vocab_size = target_vocab_size - pointer_vocab_size
        self._pointer_vocab_size = pointer_vocab_size

        # self._copy_projection_layer = nn.Linear(config.d_model,
        #                                         pointer_vocab_size)

        self._ontology_projection_layer = FeedForward(input_dim=config.d_model,
                                                      num_layers=1,
                                                      hidden_dims=self._ontology_vocab_size,
                                                      activations=LinearActivation())
        # nn.Linear(config.d_model,
        #                                         self._ontology_vocab_size,
        #                                         bias=False)

        # endregion

        # Initialize weights and apply final processing
        self.post_init()

    def _copy(self, decoder_output: torch.Tensor, encoder_output: torch.Tensor):
        single_decoding_step = False

        if len(decoder_output.shape) == 2:
            single_decoding_step = True
            decoder_output = decoder_output.unsqueeze(1)
        # The decoder_output shape: (batch_size, num_decoding_steps, embedding_dim)
        # The target_embedding shape: (batch_size, target_sequence_length, embedding_dim)
        #
        # The attention module takes vectors of shape: a=(batch_size, embedding_dim)
        # and b=(batch_size, num_rows, embedding_dim). num_rows in our example is 'target_sequence_length'.

        if not self._copy_attention:
            self._copy_attention = BilinearAttentionCuda(vector_dim=decoder_output.shape[-1],
                                                         matrix_dim=encoder_output.shape[-1],
                                                         normalize=False)

        # We want to get an attention score between the decoder_output and the target_embedding.
        #
        # We reshape the decoder_output tensor to shape (batch_size * num_decoding_steps, embedding_dim). This
        # can be viewed as each decoding step being a different batch.
        batch_size, num_decoding_steps, decoder_dim = decoder_output.size()
        a = decoder_output.reshape(batch_size * num_decoding_steps, decoder_dim)

        # We multiply the target_embedding to match the decoder_output shape. Each 'batch' need to be multiplied
        # with the same target_embedding tensor so we 'repeat' the sequence in the 'batch' dimension.
        batch_size, target_seq_length, target_dim = encoder_output.size()

        # target_embedding shape: (batch_size, target_sequence_length, embedding_dim)
        # next step shape: (batch_size, 1, target_sequence_length, embedding_dim)
        # We add another dimension so we can repeat each batch one after the other -
        # We want batch1, batch1, batch2, batch2 and not batch1, batch2, batch1, batch2
        # In the next step we repeat each batch sequence 'num_decoding_steps' times
        # Then we squeeze the batch dimension so we have each batch one after the other and we get
        # (batch_size * num_decoding_steps, target_seq_length, target_dim)
        b = encoder_output \
            .unsqueeze(1) \
            .repeat(1, num_decoding_steps, 1, 1) \
            .reshape(1, batch_size * num_decoding_steps, target_seq_length, target_dim) \
            .squeeze(0).to(device=self.device)

        # we reshape the attention score, so it match the standard shape
        copy_score = self._copy_attention(a, b).reshape(batch_size, num_decoding_steps, target_seq_length)

        if single_decoding_step:
            assert copy_score.shape[1] == 1
            copy_score = copy_score.squeeze(1)

        return copy_score

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = labels  # self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # lm_logits = self.lm_head(sequence_output)

        # region copynet
        encoder_output = hidden_states
        decoder_output = sequence_output
        copy_logits = self._copy(decoder_output, encoder_output)
        # there are 200 possible pointes in the vocab. THe copy logits are seq_length. padding the rest with 0
        padding_size = self._pointer_vocab_size - copy_logits.shape[-1]
        copy_logits = torch.nn.functional.pad(input=copy_logits, pad=(0, padding_size, 0, 0, 0, 0),
                                              mode='constant', value=0)

        logits = self._ontology_projection_layer(decoder_output)
        lm_logits = torch.cat([logits, copy_logits], dim=2)

        # endregion

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(lm_logits[:, :-1].contiguous().view(-1, lm_logits[:, :-1].contiguous().size(-1)),
                            labels[:, 1:].contiguous().view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            # logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class MT5ForConditionalGenerationWithCopyMechanism(T5ForConditionalGenerationWithCopyMechanism):
    r"""
    This class overrides [`T5ForConditionalGeneration`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    Examples:
    ```python
    >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer
    >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> with tokenizer.as_target_tokenizer():
    ...     labels = tokenizer(summary, return_tensors="pt")
    >>> outputs = model(**inputs, labels=labels["input_ids"])
    >>> loss = outputs.loss
    ```"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]


@Model.register("mt5_copynet")
class MT5Copynet(Model):
    def __init__(
            self,
            model_name: str,
            vocab: Vocabulary,
            target_embedder: Embedding,
            target_namespace: str,
            pointer_vocab_size: int,
            tensor_based_metric: List[Metric] = None,
            token_based_metric: List[Metric] = None,
            decoder_start_token_id: int = 0,
            pad_token_id: int = 0,  # These are both 0 in t5-(small|base|large). Go figure.
            eos_token_id: int = 1,
            beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
            indexer: PretrainedTransformerIndexer = None,
            **kwargs,
    ):
        super().__init__(vocab)

        self.t5 = MT5ForConditionalGenerationWithCopyMechanism.from_pretrained(
            model_name,
            vocab=vocab,
            target_embedder=target_embedder,
            target_namespace=target_namespace,
            pointer_vocab_size=pointer_vocab_size,
            ignore_mismatched_sizes=True
        )

        self._target_embedder = target_embedder

        self._indexer = indexer or PretrainedTransformerIndexer(model_name, namespace="tokens")

        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self._target_namespace = target_namespace

        self._decoder_start_id = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._pad_id = self.vocab.get_token_index("@@PADDING@@", target_namespace)
        self._start_id = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_id = self.vocab.get_token_index(END_SYMBOL, target_namespace)

        # At prediction time, we'll use a beam search to find the best target sequence.
        # For backwards compatibility, check if beam_size or max_decoding_steps were passed in as
        # kwargs. If so, update the BeamSearch object before constructing and raise a DeprecationWarning
        deprecation_warning = (
            "The parameter {} has been deprecated."
            " Provide this parameter as argument to beam_search instead."
        )
        beam_search_extras = {}
        if "beam_size" in kwargs:
            beam_search_extras["beam_size"] = kwargs["beam_size"]
            warnings.warn(deprecation_warning.format("beam_size"), DeprecationWarning)
        if "max_decoding_steps" in kwargs:
            beam_search_extras["max_steps"] = kwargs["max_decoding_steps"]
            warnings.warn(deprecation_warning.format("max_decoding_steps"), DeprecationWarning)
        self._beam_search = beam_search.construct(
            end_index=self._end_id, vocab=self.vocab, **beam_search_extras
        )

        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

    def forward(
            self, source_tokens: TextFieldTensors, target_tokens: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of Bart.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are stored under the `tokens` key. If no target
            tokens are given, the source tokens are shifted to the right by 1.

        # Returns

        `Dict[str, torch.Tensor]`
            During training, this dictionary contains the `decoder_logits` of shape `(batch_size,
            max_target_length, target_vocab_size)` and the `loss`. During inference, it contains `predictions`
            of shape `(batch_size, max_decoding_steps)` and `log_probabilities` of shape `(batch_size,)`.

        """
        inputs = source_tokens
        # targets = target_tokens
        input_ids, input_mask = inputs["tokens"]["token_ids"], inputs["tokens"]["mask"]

        outputs = {}

        # If no targets are provided, then shift input to right by 1. Bart already does this internally
        # but it does not use them for loss calculation.
        # if targets is not None:
        #     target_ids, target_mask = targets["target_tokens"]["token_ids"], targets["target_tokens"]["mask"]
        # else:
        #     target_ids = input_ids[:, 1:]
        #     target_mask = input_mask[:, 1:]

        # shape: (batch_size, max_target_sequence_length)
        target_ids = util.get_token_ids_from_text_field_tensors(target_tokens)

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self._target_embedder(target_ids)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        if self.training:
            t5_outputs: T5Output = self.t5(
                input_ids,
                attention_mask=input_mask,
                decoder_input_ids=target_ids[:, :-1].contiguous(),
                decoder_attention_mask=target_mask[:, :-1].contiguous(),
                return_dict=True,
            )

            outputs["decoder_logits"] = t5_outputs.logits

            # The BART paper mentions label smoothing of 0.1 for sequence generation tasks
            outputs["loss"] = sequence_cross_entropy_with_logits(
                # bart_outputs.logits,
                cast(torch.FloatTensor, t5_outputs.logits),
                target_ids[:, 1:].contiguous(),
                target_mask[:, 1:].contiguous(),
                # cast(torch.LongTensor, target_ids),
                # cast(torch.BoolTensor, target_mask),
                label_smoothing=0.1,
                average="token",
            )

            if torch.isnan(outputs["loss"]):
                print("here")
        else:
            # Use decoder start id and start of sentence to start decoder
            initial_decoder_ids = torch.tensor(
                [[self._decoder_start_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            inital_state = {
                "input_ids": input_ids,
                "input_mask": input_mask,
            }
            beam_result = self._beam_search.search(
                initial_decoder_ids, inital_state, self.take_step
            )

            predictions = beam_result[0]
            max_pred_indices = (
                beam_result[1].argmax(dim=-1).view(-1, 1, 1).expand(-1, -1, predictions.shape[-1])
            )
            predictions = predictions.gather(dim=1, index=max_pred_indices).squeeze(dim=1)

            outputs["predictions"] = predictions
            outputs["log_probabilities"] = (
                beam_result[1].gather(dim=-1, index=max_pred_indices[..., 0]).squeeze(dim=-1)
            )

            outputs = self.make_output_human_readable(outputs)

            targets = target_ids[:, 1:].contiguous()

            if self._tensor_based_metric is not None:
                for metric in self._tensor_based_metric:
                    metric(predictions, targets)  # type: ignore

            if self._token_based_metric is not None:
                targets_tokens = self._get_predicted_tokens(targets)
                predicted_tokens = outputs["predicted_tokens"]

                for metric in self._token_based_metric:
                    metric(predicted_tokens, targets_tokens)  # type: ignore

        return outputs

    @staticmethod
    def _decoder_cache_to_dict(decoder_cache: List[KeyValueStates]) -> Dict[str, torch.Tensor]:
        cache_dict = {}
        for layer_index, layer_cache in enumerate(decoder_cache):
            # Each layer caches the key and value tensors for its self-attention and cross-attention.
            # Hence the `layer_cache` tuple has 4 elements.
            assert len(layer_cache) == 4
            for tensor_index, tensor in enumerate(layer_cache):
                key = f"decoder_cache_{layer_index}_{tensor_index}"
                cache_dict[key] = tensor
        return cache_dict

    def _dict_to_decoder_cache(self, cache_dict: Dict[str, torch.Tensor]) -> List[KeyValueStates]:
        decoder_cache: List[KeyValueStates] = []
        for block_index in range(self.decoder.num_blocks):
            base_key = f"decoder_cache_{block_index}_"
            layer_cache = (
                cache_dict[base_key + "0"].contiguous(),
                cache_dict[base_key + "1"].contiguous(),
                cache_dict[base_key + "2"].contiguous(),
                cache_dict[base_key + "3"].contiguous(),
            )
            decoder_cache.append(layer_cache)
        return decoder_cache

    def take_step(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        # Parameters

        last_predictions : `torch.Tensor`
            The predicted token ids from the previous step. Shape: `(group_size,)`
        state : `Dict[str, torch.Tensor]`
            State required to generate next set of predictions
        step : `int`
            The time step in beam search decoding.


        # Returns

        `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`
            A tuple containing logits for the next tokens of shape `(group_size, target_vocab_size)` and
            an updated state dictionary.
        """
        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)

        if "predictions" not in state:
            state["predictions"] = last_predictions
        else:
            state["predictions"] = torch.cat((state["predictions"], last_predictions), dim=1)

        # decoder_cache = None
        # decoder_cache_dict = {
        #     k: state[k].contiguous()
        #     for k in state
        #     if k not in {"input_ids", "input_mask", "encoder_states"}
        # }
        # if len(decoder_cache_dict) != 0:
        #     decoder_cache = self._dict_to_decoder_cache(decoder_cache_dict)

        encoder_outputs = (state["encoder_states"],) if "encoder_states" in state else None
        outputs = self.t5(
            input_ids=state["input_ids"] if encoder_outputs is None else None,
            attention_mask=state["input_mask"],
            encoder_outputs=encoder_outputs,
            decoder_input_ids=state["predictions"],  # last_predictions,
            past_key_values=None,  # decoder_cache,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        log_probabilities = F.log_softmax(logits, dim=-1)

        # decoder_cache = outputs.past_key_values
        # if decoder_cache is not None:
        #     decoder_cache_dict = self._decoder_cache_to_dict(decoder_cache)
        #     state.update(decoder_cache_dict)

        state["encoder_states"] = outputs.encoder_last_hidden_state

        predictions = logits.max(dim=1)[1].unsqueeze(-1)

        return log_probabilities, state

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """

        # Parameters

        output_dict : `Dict[str, torch.Tensor]`
            A dictionary containing a batch of predictions with key `predictions`. The tensor should have
            shape `(batch_size, max_sequence_length)`

        # Returns

        `Dict[str, Any]`
            Original `output_dict` with an additional `predicted_tokens` key that maps to a list of lists of
            tokens.

        """
        predictions = output_dict["predictions"]
        output_dict["predicted_tokens"] = self._get_predicted_tokens(predictions) # type: ignore
        output_dict["predicted_text"] = torch.as_tensor([" ".join(tok_list) for tok_list in output_dict["predicted_tokens"]])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                for metric in self._tensor_based_metric:
                    all_metrics.update(
                        metric.get_metric(reset=reset)  # type: ignore
                    )
            if self._token_based_metric is not None:
                for metric in self._token_based_metric:
                    all_metrics.update(metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    def _get_predicted_tokens(
        self,
        predicted_indices: torch.Tensor,
    ) -> List[List[str]]:
        """
        Convert predicted indices into tokens.
        """
        predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[List[str]] = []

        for indices in predicted_indices:
            tokens: List[str] = []
            indices = list(indices)
            if self._end_id in indices:
                indices = indices[: indices.index(self._end_id)]
            for index in indices:
                token = self.vocab.get_token_from_index(index, self._target_namespace)
                tokens.append(token)
            predicted_tokens.append(tokens)
        return predicted_tokens

    default_predictor = "seq2seq"
