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

export metric_1="attachment_scores_seq2seq"
export metric_2=null
export validation_metric="+LAS"
export model_name="xlm-roberta-large"
export config_file="seq2seq_transformer_ud_try_2.jsonnet"

dataset_dir=experiments/processed_datasets/ud/seq2seq/
# Standard Order Model
export train_data_path="$dataset_dir"/standard/en_ewt-ud-train.conllu.tsv
export valid_data_path="$dataset_dir"/standard/en_ewt-ud-dev.conllu.tsv
export test_data_path=null

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

sbatch $sbatch_params -J train_ud_seq2seq experiments/scripts/train_scripts/train_subscript.sh

exit

# Reordered Models
combined_postfixes=("" "_combined")
languages=(hindi korean persian spanish thai)
algo_arr=(HUJI RASOOLINI)
for combined_postfix in "${combined_postfixes[@]}"
do
  for lang in "${languages[@]}"
  do
    subdir=english_reordered_by_"$lang"
    for algo in "${algo_arr[@]}"
    do
      export train_data_path="$dataset_dir"/"$subdir"/english_en_ewt-ud-train_reordered_by_"$lang"_"$algo""$combined_postfix".conllu.tsv
      export valid_data_path=null
      export test_data_path="$dataset_dir"/"$subdir"/english_en_ewt-ud-dev_reordered_by_"$lang"_"$algo""$combined_postfix".conllu.tsv

      export serialization_dir="$DIR"/english_reordered_by_"$lang"_"$algo""$combined_postfix"/model_"$MODEL_IDX"/

      if [ ! -d "$serialization_dir" ]; then
        echo "$serialization_dir" does not exists. Creating...
        mkdir -p "$serialization_dir"
      fi

      sbatch $sbatch_params -J train_ud_seq2seq experiments/scripts/train_scripts/train_subscript.sh

    done
  done
done


