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

export config_file="re_luke_finetune.jsonnet"
export eval_all="true"

MODEL_IDX="$EXPERIMENT_NUM"

dataset_dir=experiments/processed_datasets/indore/standard_split/
#telugu
languages=(hindi)
algo_arr=(HUJI RASOOLINI)
count_arr=( 100 200 300 )
for lang in "${languages[@]}"
do
  export train_data_path="$dataset_dir"/"$lang"_indore_train.tsv.json
  export valid_data_path=null
  export test_data_path="$dataset_dir"/"$lang"_indore_test.tsv.json
  export model_archive="$DIR"/english_standard/model_"$MODEL_IDX"/model.tar.gz
  for examples_count in "${count_arr[@]}"
  do
    export examples_count="$examples_count"
    export serialization_dir="$DIR"/finetuned/english_standard_finetuned_"$lang"_"$examples_count"/model_"$MODEL_IDX"/
    if [ ! -d "$serialization_dir" ]; then
      echo "$serialization_dir" does not exists. Creating...
      mkdir -p "$serialization_dir"
    fi

    sbatch $sbatch_params -J ft_indore experiments/scripts/train_scripts/finetune_scripts/finetune_subscript.sh
  done
done

# combined_postfixes=("" "_combined")
combined_postfixes=("_combined")
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    export train_data_path="$dataset_dir"/"$lang"_indore_train.tsv.json
    export valid_data_path=null
    export test_data_path="$dataset_dir"/"$lang"_indore_test.tsv.json

    for algo in "${algo_arr[@]}"
    do
      export model_archive="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/model.tar.gz
      for examples_count in "${count_arr[@]}"
      do
        export examples_count="$examples_count"
        export serialization_dir="$DIR"/finetuned/english_reordered_by_"$lang"_"$algo""$combined_postfix"_finetuned_"$examples_count"/model_"$MODEL_IDX"/
        if [ ! -d "$serialization_dir" ]; then
          echo "$serialization_dir" does not exists. Creating...
          mkdir -p "$serialization_dir"
        fi

        sbatch $sbatch_params -J ft_indore experiments/scripts/train_scripts/finetune_scripts/finetune_subscript.sh
      done
    done
  done
done