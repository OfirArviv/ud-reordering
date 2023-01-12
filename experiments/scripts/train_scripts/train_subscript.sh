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

metrics_file="$serialization_dir"/"metrics.json"
if [ -f "$metrics_file" ]
then
  echo "exit: ""$metrics_file"
  exit 0
fi

best_model_file="$serialization_dir"/"best.th"
if [ -f "$best_model_file" ]
then
  exit
  echo "recover: ""$best_model_file"
  allennlp train "$serialization_dir"/config.json --recover --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging
else
  echo "new: ""$best_model_file"

  allennlp train "$PWD"/experiments/train_configs/"$config_file" --serialization-dir "$serialization_dir" --include-package experiments.allennlp_extensions --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'
fi



