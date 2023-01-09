import argparse
import glob
import os.path


def run_model_evaluation(main_models_dir: str, expected_models_count: int):
    model = main_models_dir

    assert os.path.isdir(model)
    while model[-1] in ["/", "\\"]:
        model = model[:-1]

    existing_models = []
    for model_idx_path in glob.glob(f'{model}/*'):
        if not os.path.exists(f'{model_idx_path}/model.tar.gz'):
            existing_models.append(model_idx_path)

    existing_models_idx = [os.path.basename(p) for p in existing_models]
    print(existing_models_idx)
    if set(existing_models_idx) != set(range(1, expected_models_count + 1)):
        missing_models = set(range(1, expected_models_count + 1)).difference(set(existing_models_idx))
        print(f'{main_models_dir} is missing the following models: {missing_models}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-m", "--model-dir", required=True)
    argparser.add_argument("-c", "--expected-models-count", type=int, required=True)

    args = argparser.parse_args()

    run_model_evaluation(args.model_dir, args.expected_models_count)
