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

best_model_file="$serialization_dir"/"metrics.json"
if [ -f "$metrics_file" ]
then
  exit
fi

best_model_file="$serialization_dir"/"best.th"
if [ -f "$best_model_file" ]
then
  allennlp train "$serialization_dir"/config.json --recover --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging
else
  allennlp train "$PWD"/experiments/train_configs/copynet_transformer.jsonnet --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'
fi



