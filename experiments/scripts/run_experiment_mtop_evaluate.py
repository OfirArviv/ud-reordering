import argparse
import glob
import os.path


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
            output_file_path = f'{output_dir}/{model_basename}/{dataset_name}.json'

            for model_idx_path in glob.glob(f'{model}/*'):
                print(f'Evaluating model: {model_idx_path}, test file: {test_file}, output_path: {output_file_path}')
                allennllp_evaluate(f'{model_idx_path}/model.tar.gz', test_file, output_file_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-d", "--model-dir", required=True)

    args = argparser.parse_args()

    run_evaluation_pointer_format(args.model_dir)
