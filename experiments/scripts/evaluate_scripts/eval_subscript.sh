#!/usr/bin/env bash
#SBATCH --mem=12G
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

extra_args=""
if [ $eval_all=="true" ]
then
  echo "extra_args"
  extra_args="--eval-on-all"
fi


echo python experiments/scripts/evaluate_scripts/evaluate_base.py -m "$model_dir" -o "$output_dir" -t "$test_dir" $extra_args




