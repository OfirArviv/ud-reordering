import argparse
import glob
import json
import os.path
import statistics


def run_evaluation_pointer_format(main_models_dir: str, output_dir: str, test_files_dir: str):
    from experiments.scripts.allennlp_evaluate_custom import allennllp_evaluate

    os.makedirs(output_dir, exist_ok=True)
    sub_models_dir_list = glob.glob(f'{main_models_dir}/*')

    for model in sub_models_dir_list:
        assert os.path.isdir(model)
        model_basename = os.path.basename(model)

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

            metrics_list = []
            for metric_path in glob.glob(f'{metric_output_dir}/*.json'):
                with open(metric_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    metrics_list.append(json_data)

            agg_metric = {
                "model_count": len(metrics_list)
            }

            for k in metrics_list[0].keys():
                temp_val_list = [d[k] for d in metrics_list]
                metric_dict = {
                    f'{k}_mean': statistics.mean(temp_val_list),
                    f'{k}_std': statistics.stdev(temp_val_list)
                }
                agg_metric.update(metric_dict)

            with open(f'{metric_output_dir}/{dataset_name}_agg.json', 'x', encoding='utf-8') as f:
                json.dump(agg_metric, f)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-m", "--model-dir", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)
    argparser.add_argument("-t", "--test-dir", required=True)

    args = argparser.parse_args()

    run_evaluation_pointer_format(args.model_dir, args.output_dir, args.test_dir)
