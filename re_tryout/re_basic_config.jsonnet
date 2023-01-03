local transformers_model_name="xlm-roberta-large";
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local extra_tokens =  ["<ent>", "<ent2>"];
{
  "dataset_reader": {
    "type": "re_basic",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformers_model_name,
        "tokenizer_kwargs": {
          "additional_special_tokens": extra_tokens
        }
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformers_model_name,
      "add_special_tokens": true,
      "tokenizer_kwargs": {
        "additional_special_tokens": extra_tokens
      }
    }
  },
  "train_data_path": train_data_path,
  "validation_data_path": valid_data_path,
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformers_model_name,
          "train_parameters": true,
          "tokenizer_kwargs": {
            "additional_special_tokens": extra_tokens
          }
        }
      }
    },
    "seq2vec_encoder": {
      "type": "bert_pooler",
      "pretrained_model": transformers_model_name,
      "dropout": 0.1
    },
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": [
        "tokens"
      ],
      "batch_size": 32
    }
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
      "weight_decay": 0.01
    },
    "checkpointer": {
      "keep_most_recent_by_count": 1
    }
  }
}