import argparse
import glob
import json
import os.path
from collections import defaultdict
import pandas as pd


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


def run_agg_evaluation(main_models_dir: str, output_dir: str):
    print(f'main_models_dir:{main_models_dir}')

    # Dict of metric -> dataframe
    val_dict = defaultdict(pd.DataFrame)
    model_type = glob.glob(f'{main_models_dir}/*/')
    model_count = None
    for model in model_type:
        print(f'model type: {model}')
        model_basename = os.path.basename(model.strip("\\").strip("/"))

        if "reordered" in model_basename:
            model_lang = model_basename.split("english_reordered_by_")[1].split("_")[0]
        else:
            model_lang = "english"

        if "HUJI_RASOOLINI" in model_basename:
            print(f'Skipping {model}')
            continue

        if "HUJI_combined" in model_basename:
            model_type = "HUJI_ENSEMBLE"
        elif "HUJI" in model_basename:
            model_type = "HUJI"
        elif "RASOOLINI_combined" in model_basename:
            model_type = "RASOOLINI_ENSEMBLE"
        elif "RASOOLINI" in model_basename:
            model_type = "RASOOLINI"
        elif "standard" in model_basename:
            model_type = "VANILLA"
        else:
            raise NotImplementedError(model_basename)

        datasets_dirs = glob.glob(f'{model}/*/')

        for dataset in datasets_dirs:
            dataset_name = os.path.basename(dataset.strip("\\").strip("/"))
            if dataset_name == "predictions":
                continue

            print(f'dataset: {dataset}')

            dataset_lang = get_lang_from_filename(dataset_name)

            if model_lang != "english" and model_lang != dataset_lang:
                print(f'model lang {model_lang} does not match dataset lang {dataset_lang}')
                print(f'model name: {model_basename}')
                print(f'dataset name: {dataset_name}')
                continue

            agg_file = glob.glob(f'{dataset}/*agg*.json')
            assert len(agg_file) == 1, f'{model_basename} - {agg_file}'
            agg_file = agg_file[0]

            with open(agg_file, 'r', encoding='utf-8') as f:
                agg_data = json.load(f)

            for metric_k, v in agg_data.items():
                if metric_k == "model_count":
                    if model_count is None:
                        model_count = v
                    else:
                        assert model_count == v, f'{model_basename} - {dataset_name}'
                else:
                    v = round(v*100, 1)
                    df = val_dict[metric_k]
                    df.loc[dataset_lang, model_type] = v

    for metric, df in val_dict.items():
        try:
            df['HUJI_ENSEMBLE-VANILLA'] = df['HUJI_ENSEMBLE'] - df['VANILLA']
            if "RASOOLINI" in df.columns:
                column_order = ['VANILLA', 'HUJI', 'HUJI_ENSEMBLE',  'RASOOLINI', 'RASOOLINI_ENSEMBLE',
                                'HUJI_ENSEMBLE-VANILLA']
            else:
                column_order = ['VANILLA', 'HUJI', 'HUJI_ENSEMBLE', 'HUJI_ENSEMBLE-VANILLA']

            df = df[column_order]

            metric = metric.replace("/", "_").replace("\\", "_")
            output_path = f'{output_dir}/{metric}_model_count_{model_count}.csv'
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_path)
        except Exception as e:
            print("-------------")
            # print(str(e))
            # print(model_basename)
            # print(model_lang)
            # print(dataset_name)
            # print(dataset_lang)
            # print(df)

if __name__ == '__main__':
    # run_agg_evaluation('experiments_results/evaluation/mtop', "a")
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-m", "--model-dir", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)

    args = argparser.parse_args()

    run_agg_evaluation(args.model_dir, args.output_dir)
