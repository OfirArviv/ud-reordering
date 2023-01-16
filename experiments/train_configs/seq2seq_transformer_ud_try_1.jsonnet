// =================== Configurable Settings ======================
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local test_data_path = std.extVar('test_data_path');
local metric_1 = std.extVar('metric_1');
local metric_2 = std.extVar('metric_2');
local validation_metric = std.extVar('validation_metric');
local model_name = std.extVar('model_name');
// ================================================================
{
  "dataset_reader": {
    "type": "seq2seq_length_filtering",
    "source_max_tokens": 100,
    "target_max_tokens": 200,
    "source_add_start_token": false,
    "source_add_end_token": false,
    "source_tokenizer": {
          "type": "pretrained_transformer",
          "model_name": model_name
        },
    "target_tokenizer": {
       "type": "whitespace"
    },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name,
        "max_length": 1024
      }
    },
    "target_token_indexers": {
      "target_tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      }
    }
  },
  "train_data_path": train_data_path,
  [if valid_data_path != "null" then "validation_data_path" else null]: valid_data_path,
  [if test_data_path != "null" then "test_data_path" else null]: test_data_path,
  [if test_data_path != "null" then "evaluate_on_test" else null]: true,
  "model": {
    "type": "composed_seq2seq",
    "source_text_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "max_length": 1024,
            "train_parameters":  true
        }
      }
    },
    "encoder": {
      "type": "pass_through",
      "input_dim": 1024
    },
    "decoder": {
      "type": "auto_regressive_seq_decoder_custom",
      "scheduled_sampling_ratio": 0,
      "beam_search": {
          "beam_size": 4,
          "max_steps": 100
      },
      "decoder_net": {
        "type": "stacked_self_attention",
        "decoding_dim": 1024,
        "target_embedding_dim": 1024,
        "feedforward_hidden_dim": 512,
        "num_layers": 4,
        "num_attention_heads": 8
      },
      "target_embedder": {
        "embedding_dim": 1024,
        "vocab_namespace": "target_tokens"
      },
      "target_namespace": "target_tokens",
      "token_based_metric": metric_1
}
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 32
    }
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "grad_norm": 5.0,
    "validation_metric": validation_metric,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
      "weight_decay": 0.01,
    },
    "checkpointer": {
      "keep_most_recent_by_count": 1
    }
  }
}
