#!/usr/bin/env bash

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export action="train"
export model_id="facebook/xglm-7.5B"
export train_dataset_path="experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv"
export dev_dataset_path="experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv"
export output_dir="temp_output_xglm_mtop"
export use_lora=true
export use_qlora=true


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch  -J hf_mtop experiments/hugginface_models/run_subscript.sh
