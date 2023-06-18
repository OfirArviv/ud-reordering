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

extra_params=""
if [ "$use_lora" ]
 then
   extra_params="$extra_params"" --lora"
fi

if [ "$use_qlora" ]
 then
   extra_params="$extra_params"" --qlora"
fi

if [ "$add_instruct" ]
 then
   extra_params="$extra_params"" --add-instruction"
fi

if [ "$max_length" ]
 then
   extra_params="$extra_params"" --max-length"" $max_length"
fi

if [ "$dev_dataset_path" ]
 then
   extra_params="$extra_params"" --dev-dataset-path ""$dev_dataset_path"
fi

python "$PWD"/experiments/hugginface_models/run.py train --model-id  "$model_id" --train-dataset-path "$train_dataset_path" --dev-dataset-path "$dev_dataset_path" --output-dir "$output_dir" --seed "$RANDOM" $extra_params


