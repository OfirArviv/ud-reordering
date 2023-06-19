#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-m', '--dir', required=True)
parser.add_argument('-o', '--output-dir', required=True)
parser.add_argument('-k', '--killable', action='store_true', default=False)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

if [ "$KILLABLE" ]
 then
   sbatch_params="--killable --requeue"
  else
    sbatch_params=""
fi


export test_dir="experiments/processed_datasets/mtop/non_pointer_format/standard/"
export output_dir=$OUTPUT_DIR

count_arr=( 100 200 300)
languages=(hindi thai french spanish german)

for lang in "${languages[@]}"
do
  for examples_count in "${count_arr[@]}"
  do
    export examples_count="$examples_count"
    export model_dir="$DIR"/finetuned/english_standard_finetuned_"$lang"_"$examples_count"/

    sbatch $sbatch_params -J eval_mtop experiments/scripts/evaluate_scripts/finetune_evaluation/eval_subscript_ft.sh
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
        sbatch $sbatch_params -J eval_mtop experiments/scripts/evaluate_scripts/finetune_evaluation/eval_subscript_ft.sh
      done
    done
  done
done

