from transformers import Seq2SeqTrainer
import os
import logging

logger = logging.getLogger(__name__)
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union


class CausalTrainer(Seq2SeqTrainer):

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        gen_kwargs['use_cache'] = self.model.config.use_cache
        eval_returned = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        self.state.save_to_json(os.path.join(self.args.output_dir, 'trainer_state.json'))
        logger.warning(f"Saved trainer state to {os.path.join(self.args.output_dir, 'trainer_state.json')}")
        return eval_returned

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        # default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        # gen_kwargs["synced_gpus"] = (
        # gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        # )

        # if "attention_mask" in inputs:
        #     gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        # if "global_attention_mask" in inputs:
        #     gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names

        # if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
        #     generation_inputs = inputs[self.model.encoder.main_input_name]
        # else:
        #     generation_inputs = inputs[self.model.main_input_name]

        ids = inputs.pop("__id__") if "__id__" in inputs else None

        # inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        # cut inputs['decoder_input_ids'] and leave only the first token (bos)
        if self.model.config.is_encoder_decoder:
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
            # inputs['decoder_input_ids'] = inputs['decoder_input_ids'][:, :1]
            generated_tokens = self.model.generate(**inputs, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id)
            new_labels = inputs["labels"]
        else:
            # logger.warning('replaceing max_length with max_new_tokens')
            gen_kwargs['max_new_tokens'] = gen_kwargs.get("max_length")
            del gen_kwargs['max_length']

            new_inputs = {'input_ids': []}
            for i in range(len(inputs['input_ids'])):
                # if i not in [330]: continue

                # find the first label that is not -100
                first_target_idx = (inputs["labels"][i] != -100).nonzero(as_tuple=False).min().item()
                new_inputs['input_ids'].append(inputs['input_ids'][i][:first_target_idx])

            longest_input = max([len(x) for x in new_inputs['input_ids']])
            assert longest_input + gen_kwargs['max_new_tokens'] < self.tokenizer.model_max_length, \
                f"generation_max_length + input_source_length >= " \
                f"{longest_input + gen_kwargs['max_new_tokens']}\n" \
                f"while the model maximum position embedding length is {self.tokenizer.model_max_length}." \
                f"\nthis means that an error will be raised by the model because\n you are trying to input a length larget than the " \
                f"model maximum position embedding. Reduce the generation length or the max_source_length." \
                f"\n\n(if you didn't set generation_max_length, it defaults to val_max_target_length that itself defaults to 128)"

            new_inputs = self.tokenizer.pad(new_inputs, return_tensors="pt").to(self.model.device)
            generated_tokens = self.model.generate(**new_inputs, **gen_kwargs,  pad_token_id=self.tokenizer.eos_token_id)

            new_generated_tokens = []
            new_labels = []
            for i in range(len(generated_tokens)):
                first_target_idx = (inputs["labels"][i] != -100).nonzero(as_tuple=False).min().item()
                last_target_idx = (inputs["labels"][i] != -100).nonzero(as_tuple=False).max().item()
                last_input_idx = new_inputs['input_ids'][i].shape[-1]

                new_generated_tokens.append(generated_tokens[i][last_input_idx:])
                new_labels.append(inputs['labels'][i][first_target_idx:last_target_idx + 1])

            # pad all the generated tokens to the same length
            generated_tokens = self.tokenizer.pad({"input_ids": new_generated_tokens}, return_tensors="pt")["input_ids"]
            # pad all the labels to the same length, it is to add a pad token here, it will be ignored when detokenizin
            new_labels = self.tokenizer.pad({"input_ids": new_labels}, return_tensors="pt")["input_ids"]

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = new_labels
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                    gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        if ids is not None:
            labels = ids

        return (loss, generated_tokens, labels)