// =================== Configurable Settings ======================
local train_data_path = std.extVar('train_data_path');
local valid_data_path = std.extVar('valid_data_path');
local test_data_path = std.extVar('test_data_path');
local model_name = std.extVar('model_name');
// ================================================================
{
   "vocabulary": {
    "type": "from_files",
    "directory": "experiments/vocabs/ud"
  },
    "dataset_reader":{
      "type": "universal_dependencies_custom",
      "token_indexers": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": model_name,
          "max_length": 1024
        }
      },
    },
    "train_data_path": train_data_path,
    [if valid_data_path != "null" then "validation_data_path" else null]: valid_data_path,
    [if test_data_path != "null" then "test_data_path" else null]: test_data_path,
    [if test_data_path != "null" then "evaluate_on_test" else null]: true,
    "model": {
      "type": "biaffine_parser_custom",
      "metric": "attachment",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": model_name,
            "max_length": 1024,
            "train_parameters":  true
          }
        }
      }
      "encoder": {
        "type": "pass_through",
        "input_dim": 1024
      }
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
    },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size":128
    }
  },
    "trainer": {
      "num_epochs": 100,
      "patience": 10,
      "grad_norm": 5.0,
      "validation_metric": "+LAS",
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
        "keep_most_recent_by_count": 1
      }
    }
}