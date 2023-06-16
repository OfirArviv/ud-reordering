#!/usr/bin/env bash
#SBATCH --mem=124
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv_hf/bin/activate

module load cuda/11.7
export PYTHONPATH=.:$PYTHONPATH
python "$PWD"/experiments/hugginface_models/train_model.py


