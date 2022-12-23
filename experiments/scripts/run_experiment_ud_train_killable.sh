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

export vocab_path="experiments/vocabs/ud"
export metric_1="attachment_scores_seq2seq"
export metric_2=null
export validation_metric="+LAS"
export model_name="xlm-roberta-large"
export pointer_vocab_size=0

dataset_dir=experiments/processed_datasets/ud/
# Standard Order Model
export train_data_path="$dataset_dir"/seq2seq_standard/en_ewt-ud-train.tsv
export valid_data_path=null
export test_data_path="$dataset_dir"/seq2seq_standard/en_ewt-ud-dev.tsv

export serialization_dir="$DIR"/english_standard/model_"$MODEL_IDX"/
if [ ! -d "$serialization_dir" ]; then
 echo "$serialization_dir" does not exists. Creating...
 mkdir -p "$serialization_dir"
fi

sbatch --killable --requeue experiments/scripts/run_experiment_train_subscript.sh

