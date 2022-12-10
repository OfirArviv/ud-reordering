import json
import os


def f(dataset_path: str, output_dir: str):
    with open(dataset_path, 'r', encoding='utf-8') as i_f, \
            open(f'{output_dir}/{os.path.basename(dataset_path)}', 'x', encoding='utf-8') as o_f:
        for l in i_f:
            source, target = l.strip('\n').split('\t')
            dict = {
                "text": source,
                "summary": target
            }

            o_f.write(f'{json.dumps(dict)}\n')


f("experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv",
  "hugginface_summ/dataset")
f("experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv",
  "hugginface_summ/dataset")
f("experiments/processed_datasets/mtop/non_pointer_format/standard/hi_test_decoupled_format.tsv",
  "hugginface_summ/dataset")