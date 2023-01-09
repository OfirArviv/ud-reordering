#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-m', '--model-dir', required=True)
parser.add_argument('-c', '--count', required=True)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

export expected_models_count="$COUNT"

# Standard Order Model
export model_dir="$DIR"/english_standard/

python experiments/scripts/validate_script/validate_base.py -m "$model_dir" -c "$expected_models_count"

# Reordered Models
combined_postfixes=("" "_combined")
languages=(japanese italian)
algo_arr=(HUJI RASOOLINI HUJI_RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    for algo in "${algo_arr[@]}"
    do

      export model_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/

      python experiments/scripts/validate_script/validate_base.py -m "$model_dir" -c "$expected_models_count"

    done
  done
done

