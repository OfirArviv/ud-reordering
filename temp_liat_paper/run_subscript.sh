#!/usr/bin/env bash
#SBATCH --mem=32G
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c2

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv_hf/bin/activate

python "$PWD"/temp_liat_paper/run.py --output-dir "$output_dir" --dataset-id "$dataset_id" --seed "$RANDOM"


