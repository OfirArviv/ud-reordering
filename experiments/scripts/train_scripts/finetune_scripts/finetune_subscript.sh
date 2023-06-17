#!/usr/bin/env bash
#SBATCH --mem=32G
#SBATCH --time=2-0
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

rm -rf "$serialization_dir"/*
exit 0
allennlp train "$PWD"/experiments/train_configs/"$config_file" --serialization-dir "$serialization_dir" --include-package experiments.allennlp_extensions --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'



