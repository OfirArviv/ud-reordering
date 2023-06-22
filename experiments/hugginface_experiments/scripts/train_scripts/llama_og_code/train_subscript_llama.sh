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


python "$PWD"/experiments/hugginface_experiments/src/train_llama.py train --train-dataset-path "$train_dataset_path"  --output-dir "$output_dir" --eval-dataset-path "$eval_dataset_path"

