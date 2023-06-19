from transformers import set_seed

from experiments.hugginface_models.run import load_mtop_dataset, train_model, load_nli_dataset, evaluate_model

def evaluate():
    set_seed(42)
    model_path = "debug_models/checkpoint-6240"
    eval_dataset_path = "experiments/processed_datasets/xnli/english_reordered_by_hindi/english_xnli_eval_reordered_by_hindi_HUJI.csv"
    eval_dataset = load_nli_dataset(eval_dataset_path, False)
    output_dir = "temp"

    evaluate_model(model_id=model_path,
                   is_seq2seq_model=True,
                   train_with_lora=False,
                   train_in_4_bit=False,
                   eval_dataset=eval_dataset,
                   output_dir=output_dir,
                   cache_dir=None,
                   label="nli_test",
                   max_length=200)


if __name__ == '__main__':
    evaluate()
