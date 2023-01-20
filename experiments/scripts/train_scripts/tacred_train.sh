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

export config_file="re_luke.jsonnet"
MODEL_IDX="$EXPERIMENT_NUM"

dataset_dir=experiments/processed_datasets/tacred_small/
# Standard Order Model
export train_data_path="$dataset_dir"/standard/train.conllu.json
export valid_data_path="$dataset_dir"/standard/dev.conllu.json

export serialization_dir="$DIR"/english_standard/model_"$MODEL_IDX"/
if [ ! -d "$serialization_dir" ]; then
 echo "$serialization_dir" does not exists. Creating...
 mkdir -p "$serialization_dir"
fi

if [ "$KILLABLE" ]
 then
   sbatch_params="--killable --requeue"
  else
    sbatch_params=""
fi

sbatch $sbatch_params -J train_re_luke experiments/scripts/train_scripts/train_subscript.sh

# Reordered Models
combined_postfixes=("" "_combined")
languages=(telugu) #korean russian)
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    subdir=english_reordered_by_"$lang"
    for algo in "${algo_arr[@]}"
    do
      export train_data_path="$dataset_dir"/"$subdir"/train_reordered_by_"$lang"_"$algo""$combined_postfix".tsv.json
      export valid_data_path="$dataset_dir"/"$subdir"/dev_reordered_by_"$lang"_"$algo""$combined_postfix".tsv.json

      export serialization_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/

      if [ ! -d "$serialization_dir" ]; then
        echo "$serialization_dir" does not exists. Creating...
        mkdir -p "$serialization_dir"
      fi

      sbatch $sbatch_params -J train_re_luke experiments/scripts/train_scripts/train_subscript.sh

    done
  done
done

