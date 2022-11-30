import glob
import os.path

from experiments.scripts.allennlp_evaluate import allennllp_evaluate


def main():
    main_models_dir = "experiments_results/models/mtop"
    output_dir = "experiments_results/evaluation/mtop"
    sub_models_dir_list = glob.glob(f'{main_models_dir}/')

    for model in sub_models_dir_list:
        assert os.path.isdir(model)
        model_basename = os.path.basename(model)

        test_files_dir = "experiments/processed_datasets/mtop/pointers_format/standard/"
        if model_basename == "english_standard":
            test_files = glob.glob(f'{test_files_dir}/*.tsv')
        else:
            model_name_part = model_basename.split("_")
            lang = model_name_part[3]
            lang_code: str
            if lang == "spanish":
                lang_code = "de"
            elif lang == "spanish":
                lang_code = "es"
            elif lang == "french":
                lang_code = "fr"
            elif lang == "hindi":
                lang_code = "hi"
            elif lang == "thai":
                lang_code = "th"
            else:
                raise NotImplemented(f'{model}')

            test_files = [f'{lang_code}_test_decoupled_format.tsv']

        for test_file in test_files:
            dataset_path = f'{test_files_dir}/{test_file}'
            output_file_path = f'{output_dir}/{model_basename}/{dataset_path}.json'

            allennllp_evaluate(f'{model}/model.tar.gz', dataset_path, output_file_path)

if __name__ == "__main__":
    main()
