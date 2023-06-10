// =================== Configurable Settings ======================
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local test_data_path = std.extVar('test_data_path');
local metric_1 = std.extVar('metric_1');
local metric_2 = std.extVar('metric_2');
local validation_metric = std.extVar('validation_metric');
local model_name = std.extVar('model_name');
local model_archive = std.extVar('model_archive');
local examples_count = std.extVar('examples_count');
// ================================================================
{
  "dataset_reader": {
    "type": "seq2seq_length_filtering",
    "max_examples": examples_count,
    "source_max_tokens": 100,
    "target_max_tokens": 200,
    "source_add_start_token": false,
    "source_add_end_token": false,
    "source_tokenizer": {
          "type": "pretrained_transformer",
          "model_name": model_name
        },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name,
        "max_length": 1024
      }
    }
  },
    "validation_dataset_reader": {
    "type": "seq2seq_length_filtering",
    "source_max_tokens": 100,
    "target_max_tokens": 200,
    "source_add_start_token": false,
    "source_add_end_token": false,
    "source_tokenizer": {
          "type": "pretrained_transformer",
          "model_name": model_name
        },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name,
        "max_length": 1024
      }
    }
  },
  "train_data_path": train_data_path,
  [if valid_data_path != "null" then "validation_data_path" else null]: valid_data_path,
  [if test_data_path != "null" then "test_data_path" else null]: test_data_path,
  [if test_data_path != "null" then "evaluate_on_test" else null]: true,
"model": {
    "type": "from_archive",
    "archive_file": model_archive,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size":8
    }
  },
  "trainer": {
    "num_epochs": 50,
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
