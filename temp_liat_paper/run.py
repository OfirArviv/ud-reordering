import argparse
import os

from datasets import load_dataset

from experiments.hugginface_models.run import train_model

if __name__ == '__main__':
    if os.path.exists('/dccstor'):
        cache_dir = '/dccstor/gmc/users/ofir.arviv/transformers_cache'
    if os.path.exists('/cs/labs/oabend'):
        cache_dir = '/cs/labs/oabend/ofir.arviv/transformers_cache'
    else:
        cache_dir = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--dataset-id', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)

    args = parser.parse_args()

    datasets_map_dict = {
        0: "temp_liat_paper/dataset/0bf63aa498",
        1: "temp_liat_paper/dataset/1c909d2615",
        2: "temp_liat_paper/dataset/a587cd42d8"
    }

    selected_dataset_dir = datasets_map_dict[args.dataset_id]

    dataset = load_dataset('csv', data_files={'train': f'{selected_dataset_dir}/train.csv',
                                              'dev': f'{selected_dataset_dir}/dev.csv'})
    train_dataset = dataset['train']
    dev_dataset = dataset['dev']

    seed = args.seed
    dataset_key = selected_dataset_dir.split("/")[-1]
    model_id = "google/flan-t5-xxl"
    output_dir = f'{args.output_dir}/{model_id.replace("/","_")}_dataset_{dataset_key}_seed_{seed}'

    train_model(model_id=model_id,
                is_seq2seq_model=True,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                output_dir=output_dir,
                train_with_lora=True,
                train_in_4_bit=True,
                cache_dir=cache_dir)
