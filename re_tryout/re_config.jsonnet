local transformers_model_name="xlm-roberta-large";
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local test_data_path = std.extVar('test_data_path');
local extra_tokens =  ["<ent>", "<ent2>"];
{
  "model":{
    "type": "relation_classifier",
      "feature_extractor": {
        "type": "token",
        "embedder": {
          "type": "pretrained_transformer",
          "model_name": transformers_model_name,
          "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
        },
        "feature_type": "entity_start"
    }
  },
  "dataset_reader": {
    "type": "relation_classification",
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
    "validation_data_path":  valid_data_path,
    "trainer": {
        "num_epochs": 100,
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
    "data_loader": {"batch_size": 8, "shuffle": true}
}