#!/usr/bin/env bash

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export action="train"
export model_id="google/mt5-base"
export train_dataset_path="experiments/processed_datasets/xnli/standard/english_xnli_train.csv"
export dev_dataset_path="experiments/processed_datasets/xnli/standard/english_xnli_eval.csv"
export output_dir="temp_outputs/mt5_xnli/en_standard"
export use_lora=0
export use_qlora=0


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J hf_mtop experiments/hugginface_models/run_subscript.sh
