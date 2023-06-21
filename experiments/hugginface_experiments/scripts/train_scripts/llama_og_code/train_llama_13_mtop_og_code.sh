#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash
argparse "$@" <<EOF || exit 1
parser.add_argument('-d', '--dir', required=True)
parser.add_argument('-i', '--model-idx', required=False)
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



dataset_dir="experiments/processed_datasets/mtop/non_pointer_format"
# Vanilla model
export train_dataset_path="$dataset_dir""/standard/english_train_decoupled_format.tsv"
export eval_dataset_path="$dataset_dir""/standard/english_eval_decoupled_format.tsv"

export output_dir="$DIR"/english_standard/model_"$MODEL_IDX"/

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J llama_mtop experiments/hugginface_experiments/scripts/train_scripts/llama_og_code/train_subscript_llama.sh

exit 0
# Reordered Models
combined_postfixes=("_combined")
# hindi thai french spanish german japanese tamil turkish
languages=(russian hungarian)
algo_arr=( HUJI )
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    subdir=english_reordered_by_"$lang"
    for algo in "${algo_arr[@]}"
    do
      export train_dataset_path="$dataset_dir"/"$subdir"/english_train_decoupled_format_reordered_by_"$lang"_"$algo""$combined_postfix".tsv
      export eval_dataset_path="$dataset_dir"/"$subdir"/english_eval_decoupled_format_reordered_by_"$lang"_"$algo""$combined_postfix".tsv

      export output_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/

      if [ ! -d "$output_dir" ]; then
        echo "$output_dir" does not exists. Creating...
        mkdir -p "$output_dir"
      fi

      sbatch $sbatch_params -J llama_mtop experiments/hugginface_experiments/scripts/train_scripts/train_subscript.sh

    done
  done
done

