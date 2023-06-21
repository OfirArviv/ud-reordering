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
export output_dir="$DIR"/"llama_2e4/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp_ experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh

exit 0

export exp_fname="run_llama_hyperparams_lr_3e5.py"
export output_dir="$DIR"/"llama_3e5/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp_ experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh


export exp_fname="run_4bit_exp_1_fp4_std_lr_1e4.py"
export output_dir="$DIR"/"exp1/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp1 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh



export exp_fname="run_4bit_exp_2_fp4_std_lr_2e4.py"
export output_dir="$DIR"/"exp2/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp2 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh



export exp_fname="run_4bit_exp_3_fp4_std_lr_3e5.py"
export output_dir="$DIR"/"exp3/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp3 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh



export exp_fname="run_4bit_exp_4_nf4_lr_3e5.py"
export output_dir="$DIR"/"exp4/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp4 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh


export exp_fname="run_4bit_exp_5_nf4_lr_1e4.py"
export output_dir="$DIR"/"exp5/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp5 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh


export exp_fname="run_4bit_exp_6_nf4_lr_2e4.py"
export output_dir="$DIR"/"exp6/"

if [ ! -d "$output_dir" ]; then
 echo "$output_dir" does not exists. Creating...
 mkdir -p "$output_dir"
fi

sbatch $sbatch_params -J exp6 experiments/hugginface_experiments/scripts/train_scripts/train_subscript_4bit_experiments.sh

