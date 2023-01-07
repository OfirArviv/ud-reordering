#!/usr/bin/env bash
#SBATCH --mem=12G
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

python experiments/scripts/evaluate_scripts/evaluate_base.py -m "$model_dir" -o "$OUTPUT_DIR" -t "$TEST_DIR"




