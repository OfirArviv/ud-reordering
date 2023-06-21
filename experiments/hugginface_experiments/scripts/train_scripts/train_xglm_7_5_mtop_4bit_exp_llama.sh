#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
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

export model_id="facebook/xglm-7.5B"
export use_lora=1
export use_qlora=1
export add_instruct=1
export max_length=200

dataset_dir="experiments/processed_datasets/mtop/non_pointer_format"
# Vanilla model
export train_dataset_path="$dataset_dir""/standard/english_train_decoupled_format.tsv"
export eval_dataset_path="$dataset_dir""/standard/english_eval_decoupled_format.tsv"
export test_dataset_path=$eval_dataset_path



export exp_fname="run_llama_hyperparams.py"
export output_dir="$DIR"/"llama/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp1 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh





