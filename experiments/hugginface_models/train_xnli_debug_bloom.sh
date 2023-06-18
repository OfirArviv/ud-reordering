#!/usr/bin/env bash

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export action="train"
export model_id="bigscience/bloom-7b1"
export train_dataset_path="experiments/processed_datasets/xnli/standard/english_xnli_train.csv"
export dev_dataset_path="experiments/processed_datasets/xnli/standard/english_xnli_eval.csv"
export output_dir="temp_outputs/bloom_xnli_tryout/en_standard_1"
export use_lora=1
export use_qlora=1
export add_instruct=1


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J bloom_xnli experiments/hugginface_models/run_subscript.sh
