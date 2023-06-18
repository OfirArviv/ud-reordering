from transformers import set_seed

from experiments.hugginface_models.run import load_mtop_dataset, train_model, load_nli_dataset, evaluate_model


def train():
    set_seed(42)
    model_id = "facebook/xglm-564M"
    train_dataset_path = "experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv"
    dev_dataset_path = "experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv"

    output_dir = "temp"

    if "mtop" in train_dataset_path:
        train_dataset = load_mtop_dataset(train_dataset_path)
        dev_dataset = load_mtop_dataset(dev_dataset_path)
    elif "xnli" in train_dataset_path:
        train_dataset = load_nli_dataset(train_dataset_path, True)
        dev_dataset = load_nli_dataset(dev_dataset_path, True)
    else:
        raise NotImplementedError(train_dataset_path)

    is_seq2seq_model = False

    # train_dataset = train_dataset.select(range(100))
    # dev_dataset = train_dataset

    train_model(model_id,
                is_seq2seq_model,
                train_dataset,
                dev_dataset,
                output_dir,
                train_with_lora=True,
                train_in_4_bit=False,
                cache_dir=None,
                max_length=248)


def evaluate():
    set_seed(42)
    model_path = "temp"
    train_dataset_path = "experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv"

    if "mtop" in train_dataset_path:
        train_dataset = load_mtop_dataset(train_dataset_path)
    elif "xnli" in train_dataset_path:
        train_dataset = load_nli_dataset(train_dataset_path)
    else:
        raise NotImplementedError(train_dataset_path)

    is_seq2seq_model = False

    output_dir = "temp"

    evaluate_model(model_id=model_path,
                   is_seq2seq_model=is_seq2seq_model,
                   train_with_lora=True,
                   train_in_4_bit=False,
                   eval_dataset=train_dataset.select(range(10)),
                   output_dir=output_dir,
                   cache_dir=None,
                   label="train_file_test",
                   max_length=248)


if __name__ == '__main__':
    train()
