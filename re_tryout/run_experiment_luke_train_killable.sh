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


dataset_dir=re_tryout/simlier_dataset_conll_format_tokenized/
# Standard Order Model
export train_data_path="$dataset_dir"/standard/en_corpora_train.tsv.json
export valid_data_path="$dataset_dir"/standard/en_corpora_test.tsv.json

export serialization_dir="$DIR"/english_standard/model_"$MODEL_IDX"/
if [ ! -d "$serialization_dir" ]; then
 echo "$serialization_dir" does not exists. Creating...
 mkdir -p "$serialization_dir"
fi

sbatch re_tryout/run_experiment_train_subscript.sh

# Reordered Models
combined_postfixes=("" "_combined")
languages=(arabic korean persian)
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    subdir=english_reordered_by_"$lang"_nyuad
    for algo in "${algo_arr[@]}"
    do
      export train_data_path="$dataset_dir"/"$subdir"/en_corpora_train_reordered_by_"$lang"_"$algo""$combined_postfix".tsv.json
      export valid_data_path="$dataset_dir"/"$subdir"/en_corpora_test_reordered_by_"$lang"_"$algo""$combined_postfix".tsv.json

      export serialization_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/

      if [ ! -d "$serialization_dir" ]; then
        echo "$serialization_dir" does not exists. Creating...
        mkdir -p "$serialization_dir"
      fi

      sbatch re_tryout/run_experiment_train_subscript.sh

    done
  done
done

