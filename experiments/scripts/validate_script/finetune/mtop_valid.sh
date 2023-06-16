#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-m', '--dir', required=True)
parser.add_argument('-c', '--count', required=True)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

export expected_models_count="$COUNT"

count_arr=( 100 200 300 400 500 800 1000)
languages=(hindi)

for lang in "${languages[@]}"
do
  for examples_count in "${count_arr[@]}"
  do
    export examples_count="$examples_count"
    export model_dir="$DIR"/finetuned/english_standard_finetuned_"$lang"_"$examples_count"/

    python experiments/scripts/validate_script/validate_base.py -m "$model_dir" -c "$expected_models_count"
  done
done

# Reordered Models
combined_postfixes=("_combined")
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    for algo in "${algo_arr[@]}"
    do
      for examples_count in "${count_arr[@]}"
      do
        export examples_count="$examples_count"
        export model_dir="$DIR"/finetuned/english_reordered_by_"$lang"_"$algo""$combined_postfix"_finetuned_"$examples_count"/
        python experiments/scripts/validate_script/validate_base.py -m "$model_dir" -c "$expected_models_count"
      done
    done
  done
done

echo "Finished validation!"
