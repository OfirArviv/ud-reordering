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


MODEL_IDX="$EXPERIMENT_NUM"

export metric_1="em_accuracy"
export metric_2=null
export validation_metric="+em_accuracy"
export model_name="google/mt5-base"

export config_file="finetune_mt5.jsonnet"

# model_archive, config_file, serialization_dir
# train data, validation data

dataset_dir=experiments/processed_datasets/mtop/non_pointer_format/

languages=(hindi)
algo_arr=(HUJI RASOOLINI)
count_arr=( "100" "200" "300" )
for lang in "${languages[@]}"
do
  export train_data_path="$dataset_dir"/standard_small/"$lang"_eval_decoupled_format.tsv
  export valid_data_path=null
  export test_data_path="$dataset_dir"/standard/"$lang"_test_decoupled_format.tsv
  export model_archive="$DIR"/english_standard/model_"$MODEL_IDX"/
  echo "aaa"
  for examples_count in "${count_array[@]}"
  do
    export examples_count="$examples_count"
    echo "$examples_count"
    export serialization_dir="$DIR"/finetuned/english_standard_finetuned_"$lang"_"$examples_count"/model_"$MODEL_IDX"/
    if [ ! -d "$serialization_dir" ]; then
      echo "$serialization_dir" does not exists. Creating...
      mkdir -p "$serialization_dir"
    fi

    sbatch $sbatch_params -J finetune_mtop experiments/scripts/train_scripts/train_subscript.sh
  done
done


combined_postfixes=("" "_combined")
languages=(hindi)
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    export train_data_path="$dataset_dir"/standard_small/"$lang"_eval_decoupled_format.tsv
    export valid_data_path=null
    export test_data_path="$dataset_dir"/standard/"$lang"_test_decoupled_format.tsv

    for algo in "${algo_arr[@]}"
    do
      export model_archive="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/
      echo "bbb"
      for examples_count in "${count_array[@]}"
      do
        export examples_count="$examples_count"
        echo "$examples_count"
        export serialization_dir="$DIR"/finetuned/english_reordered_by_"$lang"_"$algo""$combined_postfix"_finetuned_"$examples_count"/model_"$MODEL_IDX"/
        if [ ! -d "$serialization_dir" ]; then
          echo "$serialization_dir" does not exists. Creating...
          mkdir -p "$serialization_dir"
        fi

        sbatch $sbatch_params -J finetune_mtop experiments/scripts/train_scripts/train_subscript.sh
      done
    done
  done
done