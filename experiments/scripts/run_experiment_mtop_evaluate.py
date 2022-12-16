import argparse
import glob
import json
import os.path
import statistics


def run_evaluation_pointer_format(model_dir: str):
    from experiments.scripts.allennlp_evaluate_custom import allennllp_evaluate

    main_models_dir = "experiments_results/models/mtop"
    output_dir = "experiments_results/evaluation/mtop"
    os.makedirs(output_dir, exist_ok=True)
    sub_models_dir_list = glob.glob(f'{main_models_dir}/*')

    sub_models_dir_list = [model_dir]
    for model in sub_models_dir_list:
        assert os.path.isdir(model)
        model_basename = os.path.basename(model)

        test_files_dir = "experiments/processed_datasets/mtop/pointers_format/standard/"
        test_files = glob.glob(f'{test_files_dir}/*test*.tsv')
        for test_file in test_files:
            dataset_name = os.path.basename(test_file)
            metric_output_dir = f'{output_dir}/{model_basename}/{dataset_name}'
            for model_idx_path in glob.glob(f'{model}/*'):
                model_idx = os.path.basename(model_idx_path)
                os.makedirs(metric_output_dir, exist_ok=True)
                output_file_path = f'{metric_output_dir}/{model_idx}.json'

                print(f'\n------------------------------------------\n'
                      f'Evaluating model: {model_idx_path}\n'
                      f'Test file: {test_file}\n'
                      f'Output_path: {output_file_path}\n'
                      f'------------------------------------------\n')

                allennllp_evaluate(f'{model_idx_path}/model.tar.gz', test_file, output_file_path)

            metric_list = []
            for metric_path in glob.glob(f'{metric_output_dir}/*.json'):
                with open(metric_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    acc = json_data["em_accuracy"]
                    metric_list.append(acc)

            agg_metric = {
                "model_count": len(metric_list),
                "mean": statistics.mean(metric_list),
                "std": statistics.stdev(metric_list)
            }

            with open(f'{metric_output_dir}/{dataset_name}_agg.json', 'x', encoding='utf-8') as f:
                json.dump(agg_metric, f)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-m", "--model-dir", required=True)

    args = argparser.parse_args()

    run_evaluation_pointer_format(args.model_dir)
