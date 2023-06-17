
#!/usr/bin/env bash

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

export action="train"
export model_id="bigscience/bloom-7b1"
export train_dataset_path="experiments/processed_datasets/mtop/non_pointer_format/standard/english_train_decoupled_format.tsv"
export dev_dataset_path="experiments/processed_datasets/mtop/non_pointer_format/standard/english_eval_decoupled_format.tsv"
export output_dir="temp_outputs/bloom_mtop_tryout/en_standard_1"
export use_lora=1
export use_qlora=1


if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch -J xglm_mtop experiments/hugginface_models/run_subscript.sh
