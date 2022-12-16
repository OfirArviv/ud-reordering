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
  echo "exit"
  exit 0
fi

best_model_file="$serialization_dir"/"best.th"
if [ -f "$best_model_file" ]
then
  echo "recover"
  echo allennlp train "$serialization_dir"/config.json --recover --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging
else
  echo "new"
  echo allennlp train "$PWD"/experiments/train_configs/copynet_transformer.jsonnet --serialization-dir "$serialization_dir" --include-package experiments.allennlp_extensions --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'
fi



