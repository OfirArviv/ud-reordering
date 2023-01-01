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
          "train_parameters": true,
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
      "lr": 1e-5,
      "weight_decay": 0.01
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