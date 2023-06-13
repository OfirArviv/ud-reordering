local transformers_model_name="xlm-roberta-large";
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local test_data_path = std.extVar('test_data_path');
local extra_tokens =  ["<ent>", "<ent2>"];
local model_archive = std.extVar('model_archive');
local examples_count = std.parseInt(std.extVar('examples_count'));
{
"model": {
    "type": "from_archive",
    "archive_file": model_archive,
  },
  "dataset_reader": {
    "type": "relation_classification",
          "source_max_tokens": 100,
         "max_examples": examples_count,
    "tokenizer":
        {"type": "pretrained_transformer",
                   "model_name": transformers_model_name,
                   "add_special_tokens": true,
                   "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
      },
    "token_indexers": {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name,
                       "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
    }},
  },
"validation_dataset_reader": {
    "type": "relation_classification",
          "source_max_tokens": 100,
    "tokenizer":
        {"type": "pretrained_transformer",
                   "model_name": transformers_model_name,
                   "add_special_tokens": true,
                   "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
      },
    "token_indexers": {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name,
                       "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
    }},
  },
    "train_data_path": train_data_path,
  [if valid_data_path != "null" then "validation_data_path" else null]: valid_data_path,
  [if test_data_path != "null" then "test_data_path" else null]: test_data_path,
  [if test_data_path != "null" then "evaluate_on_test" else null]: true,
    "trainer": {
        "num_epochs": 50,
           "patience" : 10,
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.weight",
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
        },
        "learning_rate_scheduler": {
            "type": "custom_linear_with_warmup",
            "warmup_ratio": 0.06
        },
        "num_gradient_accumulation_steps": 8,
        "validation_metric": "+micro_fscore"
    },
    "data_loader": {"batch_size": 16, "shuffle": true}
}