#!/usr/bin/env bash
#SBATCH --mem=124g
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv_hf/bin/activate

module load cuda/11.7
export PYTHONPATH=.:$PYTHONPATH

run_params=$action "--model-id" "$model_id" "--train-dataset-path" "$train_dataset_path" "--dev-dataset-path" "$dev_dataset_path" "--output-dir" "$output_dir" "--seed" "$RANDOM"

if [ "$use_lora" ]
 then
   run_params=""$run_params"" "--lora"
fi

if [ "$use_qlora" ]
 then
   run_params=""$run_params"" "--qlora"
fi

echo "$PWD"/experiments/hugginface_models/run.py $run_params


