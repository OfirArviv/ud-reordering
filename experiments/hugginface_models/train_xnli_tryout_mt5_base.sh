#!/usr/bin/env bash

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-i', '--id', required=True)
parser.add_argument('-k', '--killable', action='store_true', default=False)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export action="train"
export model_id="google/mt5-base"
export train_dataset_path="experiments/processed_datasets/xnli/standard/english_xnli_train.csv"
export dev_dataset_path="experiments/processed_datasets/xnli/standard/english_xnli_eval.csv"
export output_dir="temp_outputs/mt5_base_xnli/en_standard_""$ID"
# export use_lora=null
# export use_qlora=null
# export add_instruct=null


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J hf_xnli experiments/hugginface_models/run_subscript.sh

export train_dataset_path="experiments/processed_datasets/xnli/english_reordered_by_hindi/english_xnli_train_reordered_by_hindi_HUJI.csv"
export dev_dataset_path="experiments/processed_datasets/xnli/english_reordered_by_hindi/english_xnli_eval_reordered_by_hindi_HUJI.csv"
export output_dir="temp_outputs/mt5_base_xnli/en_reordered_by_hindi_""$ID"
# export use_lora=null
# export use_qlora=null


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J hf_xnli experiments/hugginface_models/run_subscript.sh