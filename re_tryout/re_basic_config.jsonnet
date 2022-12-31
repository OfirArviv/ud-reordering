{
  "dataset_reader": {
    "type": "re_basic",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": "xlm-roberta-large",
        "tokenizer_kwargs": {
          "additional_special_tokens": [
            "<ent>",
            "<ent2>"
          ]
        }
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "xlm-roberta-large",
      "add_special_tokens": true,
      "tokenizer_kwargs": {
        "additional_special_tokens": [
          "<ent>",
          "<ent2>"
        ]
      }
    }
  },
  "train_data_path": "re_tryout/simlier_dataset_conll_format/standard/en_corpora_train.tsv.json",
  "validation_data_path": "re_tryout/simlier_dataset_conll_format/standard/en_corpora_test.tsv.json",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": "xlm-roberta-large",
          "train_parameters":  true,
          "tokenizer_kwargs": {
            "additional_special_tokens": [
              "<ent>",
              "<ent2>"
            ]
          }
        }
      }
    },
    "seq2vec_encoder": {
      "type": "bert_pooler",
      "pretrained_model": "xlm-roberta-large",
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
    "patience": 10,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
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