#!/usr/bin/env bash
#SBATCH --mem=32G
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c2

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-m', '--dir', required=True)
parser.add_argument('-i', '--experiment_num', required=True)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ./venv/bin/activate

MODEL_IDX="$EXPERIMENT_NUM"

datasets_dir=experiments/processed_datasets/mtop/pointers_format/
languages=(hindi thai french spanish german)

export vocab_path=vocab
export metrics_list="['em_accuracy']"
export validation_metric="+em_accuracy"
export model_name="xlm-roberta-large"

for lang in "${languages[@]}"
do
   subdir=english_reordered_by_"$lang"
   dataset_dir="$datasets_dir"/"$subdir"

   algo_arr=(HUJI RASOOLINI)
   for algo in "${algo_arr[@]}"
   do
     export train_data_path="$dataset_dir"/english_train_decoupled_format_reordered_by_"$lang"_"$algo".tsv
     export valid_data_path="$dataset_dir"/english_eval_decoupled_format_reordered_by_"$lang"_"$algo".tsv

     serialization_dir="$DIR"/reordered_by_"$lang"_"$algo"/model_"$EXPERIMENT_NUM"/

     if [ ! -d "$serialization_dir" ]; then
       echo "$serialization_dir" does not exists. Creating...
       mkdir -p "$serialization_dir"
     fi

     allennlp train "$PWD"/experiments/train_configs/copynet_transformer.jsonnet --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'

   done
done

