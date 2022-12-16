#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-d', '--dir', required=True)
parser.add_argument('-i', '--experiment-num', required=False)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME


MODEL_IDX="$EXPERIMENT_NUM"

export vocab_path="experiments/vocabs/mtop_pointers"
export metric_1="em_accuracy"
export metric_2=null
export validation_metric="+em_accuracy"
export model_name="xlm-roberta-large"
export pointer_vocab_size=100

dataset_dir=experiments/processed_datasets/mtop/pointers_format/
# Standard Order Model
export train_data_path="$dataset_dir"/standard/english_train_decoupled_format.tsv
export valid_data_path=null
export test_data_path="$dataset_dir"/standard/english_eval_decoupled_format.tsv

export serialization_dir="$DIR"/english_standard/model_"$MODEL_IDX"/
if [ ! -d "$serialization_dir" ]; then
 echo "$serialization_dir" does not exists. Creating...
 mkdir -p "$serialization_dir"
fi

sbatch --killable --requeue experiments/scripts/run_experiment_train_subscript.sh

# Reordered Models
combined_postfixes=("" "_combined")
languages=(hindi thai french spanish german)
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    subdir=english_reordered_by_"$lang"
    for algo in "${algo_arr[@]}"
    do
      export train_data_path="$dataset_dir"/"$subdir"/english_train_decoupled_format_reordered_by_"$lang"_"$algo""$combined_postfix".tsv
      export valid_data_path=null
      export test_data_path="$dataset_dir"/"$subdir"/english_eval_decoupled_format_reordered_by_"$lang"_"$algo""$combined_postfix".tsv

      export serialization_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/

      if [ ! -d "$serialization_dir" ]; then
        echo "$serialization_dir" does not exists. Creating...
        mkdir -p "$serialization_dir"
      fi

      sbatch --killable --requeue experiments/scripts/run_experiment_train_subscript.sh

    done
  done
done

