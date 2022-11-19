// =================== Configurable Settings ======================

local vocab_path = std.extVar('vocab_path');
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local metric_1 = std.extVar('metric_1');
local validation_metric = std.extVar('validation_metric');
local model_name = std.extVar('model_name');
// ================================================================
{
 "vocabulary": {
    "type": "from_files",
    "directory": vocab_path
  },
  "dataset_reader": {
    "type": "seq2seq_length_filtering",
    "source_max_tokens": 200,
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
  "validation_data_path": valid_data_path,
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
      "type": "auto_regressive_seq_decoder_copynet",
      "scheduled_sampling_ratio": 0,
      "beam_search": {
          "beam_size": 4,
          "max_steps": 200
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
      "token_based_metric": [metric_1],
      "label_smoothing_ratio": 0.1,
      "pointer_vocab_size": 100
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size":16
    }
  },
  "trainer": {
    "num_epochs": 100,
    "grad_norm": 5.0,
    "validation_metric": validation_metric,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*transformer.*embeddings.*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]0[.]|layer[.]1[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]2[.]|layer[.]3[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]4[.]|layer[.]5[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]6[.]|layer[.]7[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]8[.]|layer[.]9[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]10[.]|layer[.]11[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]12[.]|layer[.]13[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]14[.]|layer[.]15[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]16[.]|layer[.]17[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]18[.]|layer[.]19[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]20[.]|layer[.]21[.]).*"], {"lr": 1e-5}],
        [[".*transformer.*(layer[.]22[.]|layer[.]23[.]).*"], {"lr": 1e-5}],
        [[".*transformer_model.pooler*"], {"lr": 1e-5}],
        [[".*_decoder.*"], {"lr": 1e-3}]
      ]
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 100,
      "gradual_unfreezing": true
    },
    "checkpointer": {
      "keep_most_recent_by_count": 0
    }
  }
}
