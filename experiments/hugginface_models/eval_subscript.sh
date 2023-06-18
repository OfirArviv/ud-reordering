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

if [ "$seq2seq" ]
 then
   extra_params="$extra_params"" --seq2seq"
fi

if [ "$max_length" ]
 then
   extra_params="$extra_params"" --max-length"" $max_length"
fi

python "$PWD"/experiments/hugginface_models/run.py evaluate --model-id  "$model_id" --eval-dataset-path "$eval_dataset_path"  --output-dir "$output_dir"  $extra_params


