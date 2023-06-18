#!/usr/bin/env bash

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export action="train"
export model_id="facebook/xglm-7.5B"
export train_dataset_path="experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv"
export dev_dataset_path="experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv"
export output_dir="temp_outputs/xglm_7_5_mtop/en_standard_1"
export use_lora=1
export use_qlora=1
export add_instruct=1
export max_length = 2048


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J xglm_mtop experiments/hugginface_models/run_subscript.sh
