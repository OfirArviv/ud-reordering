#!/usr/bin/env bash
#SBATCH --mem=24G
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:12g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

extra_args=""
if [ $eval_all ]
then
  extra_args="--eval-on-all"
fi


python experiments/scripts/evaluate_scripts/finetune_evaluation/evaluate_base_finetuned.py -m "$model_dir" -o "$output_dir" -t "$test_dir" $extra_args




