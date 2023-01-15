#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-m', '--model-dir', required=True)
parser.add_argument('-o', '--output-dir', required=True)
parser.add_argument('-k', '--killable', action='store_true', default=False)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export model_dir="$MODEL_DIR"/english_standard/
export output_dir=$OUTPUT_DIR
export test_dir="experiments/processed_datasets/ud/seq2seq/standard/"

if [ "$KILLABLE" ]
 then
   sbatch_params="--killable --requeue"
  else
    sbatch_params=""
fi

sbatch $sbatch_params -J eval_ud experiments/scripts/evaluate_scripts/eval_subscript.sh

# Reordered Models
combined_postfixes=("" "_combined")
languages=(french german hindi korean persian spanish thai)
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    for algo in "${algo_arr[@]}"
    do
      export model_dir="$MODEL_DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/

      sbatch $sbatch_params -J eval_ud experiments/scripts/evaluate_scripts/eval_subscript.sh

    done
  done
done

