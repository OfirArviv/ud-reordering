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
export model_id="temp_outputs/mt5_base_xnli/en_standard_""$ID"
export output_dir="temp_outputs/mt5_base_xnli/en_standard_""$ID"
export eval_dataset_path="experiments/processed_datasets/xnli/standard/thai_xnli_test.csv"
# export use_lora=null
# export use_qlora=null
# export add_instruct=null
export max_length=20
export seq2seq=1


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J eval_xnli experiments/hugginface_models/eval_subscript.sh

export model_id="temp_outputs/mt5_base_xnli/en_reordered_by_thai_""$ID"
export output_dir="temp_outputs/mt5_base_xnli/en_reordered_by_thai_""$ID"
sbatch -J eval_xnli experiments/hugginface_models/eval_subscript.sh
