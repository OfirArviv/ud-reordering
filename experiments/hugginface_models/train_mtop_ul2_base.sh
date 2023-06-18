#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-d', '--dir', required=True)
parser.add_argument('-i', '--experiment-num', required=False)
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


MODEL_IDX="$EXPERIMENT_NUM"

export model_id="google/flan-ul2"
export max_length=200
export use_lora=1
export use_qlora=1
export add_instruct=1


dataset_dir=experiments/processed_datasets/mtop/non_pointer_format/
# Standard Order Model
export train_dataset_path="$dataset_dir"/standard/english_train_decoupled_format.tsv
export test_dataset_path="$dataset_dir"/standard/english_eval_decoupled_format.tsv
# export eval_data_path=

export serialization_dir="$DIR"/english_standard/model_"$MODEL_IDX"/
if [ ! -d "$serialization_dir" ]; then
 echo "$serialization_dir" does not exists. Creating...
 mkdir -p "$serialization_dir"
fi


sbatch $sbatch_params -J ul2_mtop experiments/hugginface_models/run_subscript.sh

# Reordered Models
combined_postfixes=("_combined")
# french spanish german
languages=(hindi thai )
# RASOOLINI
algo_arr=(HUJI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    subdir=english_reordered_by_"$lang"
    for algo in "${algo_arr[@]}"
    do
      export train_dataset_path="$dataset_dir"/"$subdir"/english_train_decoupled_format_reordered_by_"$lang"_"$algo""$combined_postfix".tsv
      export test_dataset_path="$dataset_dir"/"$subdir"/english_eval_decoupled_format_reordered_by_"$lang"_"$algo""$combined_postfix".tsv
      # export eval_data_path=

      export serialization_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/

      if [ ! -d "$serialization_dir" ]; then
        echo "$serialization_dir" does not exists. Creating...
        mkdir -p "$serialization_dir"
      fi

      sbatch $sbatch_params -J ul2_mtop experiments/hugginface_models/run_subscript.sh

    done
  done
done
