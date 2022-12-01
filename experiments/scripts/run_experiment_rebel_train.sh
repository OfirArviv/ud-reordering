#!/usr/bin/env bash
#SBATCH --mem=32G
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c2

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-d', '--dir', required=True)
parser.add_argument('-i', '--experiment-num', required=False)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

. ../venv/bin/activate

MODEL_IDX="$SLURM_ARRAY_TASK_ID"

export vocab_path="experiments/vocabs/rebel_pointers"
export metric_1="em_accuracy"
export metric_2="re_fscore"
export validation_metric="+f1"
export model_name="xlm-roberta-large"
export pointer_vocab_size=300

dataset_dir=experiments/processed_datasets/rebel/
# Standard Order Model
export train_data_path="$dataset_dir"/seq2seq_standard/english_train_144976.tsv
export valid_data_path="$dataset_dir"/seq2seq_standard/english_dev_2001.tsv

serialization_dir="$DIR"/english_standard/model_"$MODEL_IDX"/
if [ ! -d "$serialization_dir" ]; then
 echo "$serialization_dir" does not exists. Creating...
 mkdir -p "$serialization_dir"
fi

allennlp train experiments/train_configs/copynet_transformer.jsonnet --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'


# Reordered Models
languages=(hindi koean vietnamese)
for lang in "${languages[@]}"
do
   subdir=seq2seq_english_reordered_by_"$lang"
   algo_arr=(HUJI RASOOLINI)
   for algo in "${algo_arr[@]}"
   do
     export train_data_path="$dataset_dir"/$subdir/english_train_144976_reordered_by_"$lang"_"$algo".tsv
     export test_data_path="$dataset_dir"/$subdir/english_dev_2001_reordered_by_"$lang"_"$algo".tsv

     serialization_dir="$DIR"/english_reordered_by_"$lang"_"$algo"/model_"$MODEL_IDX"/

     if [ ! -d "$serialization_dir" ]; then
       echo "$serialization_dir" does not exists. Creating...
       mkdir -p "$serialization_dir"
     fi

     allennlp train "$PWD"/experiments/train_configs/copynet_transformer.jsonnet --serialization-dir "$serialization_dir" --include-package experiments --file-friendly-logging --overrides '{"pytorch_seed":'"$RANDOM"', "numpy_seed":'"$RANDOM"', "random_seed": '"$RANDOM"' }'

   done
done

