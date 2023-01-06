#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-d', '--dir', required=True)
parser.add_argument('-k', '--killable', action='store_true', default=False)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Standard Order Model
model_dir="$DIR"/english_standard/
test_dir="experiments/processed_datasets/multilingual_top/pointers_format/standard/"

if [ "$KILLABLE" ]
 then
   sbatch_params="--killable --requeue"
  else
    sbatch_params=""
fi

sbatch $sbatch_params python experiments/scripts/evaluate_scripts/evaluate_base.py -m "$model_dir" -o "$OUTPUT_DIR" -t "$TEST_DIR"

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

      model_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/

      sbatch $sbatch_params python experiments/scripts/evaluate_scripts/evaluate_base.py -m "$model_dir" -o "$OUTPUT_DIR" -t "$TEST_DIR"

    done
  done
done

