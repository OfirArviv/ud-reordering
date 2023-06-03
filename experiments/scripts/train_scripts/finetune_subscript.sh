#!/usr/bin/env bash
#SBATCH --mem=32G
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate

allennlp finetune -m "$model_archive" -c "$PWD"/experiments/train_configs/"$config_file" --serialization-dir "$serialization_dir" --include-package experiments.allennlp_extensions --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'



