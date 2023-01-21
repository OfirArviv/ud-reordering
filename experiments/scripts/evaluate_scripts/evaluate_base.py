import argparse
import glob
import json
import os.path
import statistics
from experiments.scripts.allennlp_predict_custom import allennllp_predict


def get_lang_from_filename(filename: str):
    if filename.startswith("en"):
        lang = "english"
    elif "english" in filename:
        lang = "english"
    elif filename.startswith("ar"):
        lang = "arabic"
    elif filename.startswith("de"):
        lang = "german"
    elif filename.startswith("es"):
        lang = "spanish"
    elif filename.startswith("fa"):
        lang = "persian"
    elif filename.startswith("fr"):
        lang = "french"
    elif filename.startswith("it"):
        lang = "italian"
    elif filename.startswith("ko"):
        lang = "korean"
    elif filename.startswith("nl"):
        lang = "dutch"
    elif filename.startswith("pl"):
        lang = "polish"
    elif filename.startswith("pt"):
        lang = "portuguese"
    elif filename.startswith("ru"):
        lang = "russian"
    elif filename.startswith("hi"):
        lang = "hindi"
    elif filename.startswith("th"):
        lang = "thai"
    elif filename.startswith("ja"):
        lang = "japanese"
    elif filename.startswith("tr"):
        lang = "turkish"
    elif filename.startswith("sv"):
        lang = "swedish"
    elif filename.startswith("fa"):
        lang = "persian"
    elif filename.startswith("id"):
        lang = "indonesian"
    elif filename.startswith("ga"):
        lang = "irish"
    elif "hindi" in filename:
        return "hindi"
    elif "telugu" in filename:
        return "telugu"
    elif "bangali" in filename:
        return "bengali"
    elif filename.startswith("_"):
        return "_"
    else:
        raise Exception(f'Unknown lang: {filename}')

    return lang


def run_model_evaluation(main_models_dir: str, output_dir: str, test_dir: str, eval_on_all_datasets: bool):
    from experiments.scripts.allennlp_evaluate_custom import allennllp_evaluate

    os.makedirs(output_dir, exist_ok=True)
    test_files = glob.glob(f'{test_dir}/*test*')

    model = main_models_dir

    print(f'model:{model}')
    assert os.path.isdir(model)
    while model[-1] in ["/", "\\"]:
        model = model[:-1]
    model_basename = os.path.basename(model)

    if "reordered" in model_basename:
        model_lang = model_basename.split("english_reordered_by_")[1].split("_")[0]
    else:
        model_lang = "english"

    print(test_files)

    for test_file in test_files:
        dataset_name = os.path.basename(test_file)
        dataset_lang = get_lang_from_filename(dataset_name)

        if not eval_on_all_datasets and model_lang != "english" and model_lang != dataset_lang:
            print(f'model lang {model_lang} does not match dataset lang {dataset_lang}')
            print(f'model name: {model_basename}')
            print(f'dataset name: {dataset_name}')
            continue

        metric_output_dir = f'{output_dir}/{model_basename}/{dataset_name}'
        for model_idx_path in glob.glob(f'{model}/*'):
            model_idx = os.path.basename(model_idx_path)

            os.makedirs(metric_output_dir, exist_ok=True)
            output_file_path = f'{metric_output_dir}/{model_idx}.json'

            prediction_output_dir = f'{output_dir}/{model_basename}/predictions/{model_idx}'
            os.makedirs(prediction_output_dir, exist_ok=True)
            prediction_output_file = f'{prediction_output_dir}/{dataset_name}'
            print(f'\n------------------------------------------\n'
                  f'Evaluating model: {model_idx_path}\n'
                  f'Test file: {test_file}\n'
                  f'Prediction Output File: {prediction_output_file}\n'
                  f'Output_path: {output_file_path}\n'
                  f'------------------------------------------\n')

            try:
                if not os.path.exists(output_file_path):
                    allennllp_evaluate(f'{model_idx_path}/model.tar.gz', test_file, output_file_path)
                if not os.path.exists(prediction_output_file):
                    allennllp_predict(f'{model_idx_path}/model.tar.gz', test_file, prediction_output_file)
            except Exception as e:
                print(e)

        metrics_list = []
        for metric_path in glob.glob(f'{metric_output_dir}/*.json'):
            if "_agg" in metric_path:
                continue
            with open(metric_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                metrics_list.append(json_data)

        agg_metric = {
            "model_count": len(metrics_list)
        }

        for k in metrics_list[0].keys():
            temp_val_list = [d[k] for d in metrics_list]
            if len(temp_val_list) > 1:
                metric_dict = {
                    f'{k}_mean': statistics.mean(temp_val_list),
                    f'{k}_std': statistics.stdev(temp_val_list)
                }
            else:
                metric_dict = {
                    f'{k}_mean': temp_val_list[0],
                    f'{k}_std': 0
                }
            agg_metric.update(metric_dict)

        with open(f'{metric_output_dir}/{dataset_name}_agg.json', 'w', encoding='utf-8') as f:
            json.dump(agg_metric, f, indent=4)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-m", "--model-dir", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)
    argparser.add_argument("-t", "--test-dir", required=True)
    argparser.add_argument("-a", "--eval-on-all", action='store_true', default=False)

    args = argparser.parse_args()
    print(args.eval_on_all)
    exit()
    # run_model_evaluation(args.model_dir, args.output_dir, args.test_dir, args.eval_on_all)
