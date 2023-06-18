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
export max_length=200


export train_dataset_path="experiments/processed_datasets/amazon_reviews/standard/english_amazon_reviews_train.csv"
export dev_dataset_path="experiments/processed_datasets/amazon_reviews/standard/english_amazon_reviews_eval.csv"
export output_dir="temp_outputs/mt5_base_amazon_reviews/en_standard_""$ID"
# export use_lora=null
# export use_qlora=null
# export add_instruct=null

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J hf_sent experiments/hugginface_models/run_subscript.sh

export train_dataset_path="experiments/processed_datasets/amazon_reviews/english_reordered_by_japanese/english_amazon_reviews_train_reordered_by_japanese_HUJI.csv"
export dev_dataset_path="experiments/processed_datasets/amazon_reviews/english_reordered_by_japanese/english_amazon_reviews_eval_reordered_by_japanese_HUJI.csv"
export output_dir="temp_outputs/mt5_base_amazon_reviews/en_reordered_by_japanese_""$ID"
# export use_lora=null
# export use_qlora=null
# export add_instruct=null


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J hf_sent experiments/hugginface_models/run_subscript.sh

export train_dataset_path="experiments/processed_datasets/amazon_reviews/english_reordered_by_spanish/english_amazon_reviews_train_reordered_by_spanish_HUJI.csv"
export dev_dataset_path="experiments/processed_datasets/amazon_reviews/english_reordered_by_spanish/english_amazon_reviews_eval_reordered_by_spanish_HUJI.csv"
export output_dir="temp_outputs/mt5_base_amazon_reviews/en_reordered_by_spanish_""$ID"
# export use_lora=null
# export use_qlora=null
# export add_instruct=null


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J hf_sent experiments/hugginface_models/run_subscript.sh
