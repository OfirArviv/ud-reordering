#!/usr/bin/env bash
#SBATCH --mem=12G
#SBATCH --time=1-0
#SBATCH -c10

ARGPARSE_DESCRIPTION="Sample script description"      # this is optional
source /cs/labs/oabend/ofir.arviv/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-d', '--dir', required=True)

EOF

HOME="/cs/labs/oabend/ofir.arviv/"
TMP=$TMPDIR
TEMP=$TMPDIR
export HOME

for type_dir in "$DIR"/*/     # list directories in the form "/tmp/dirname/"
do
  for idx_dir in "$type_dir"/*/     # list directories in the form "/tmp/dirname/"
  do
    idx_dir=${idx_dir%*/}      # remove the trailing "/"
    echo "Working on ""${idx_dir}"
    weights_file="$idx_dir"/weights.th
    if [ ! -f "$weights_file" ]
    then
      tar xvzf "$idx_dir"/model.tar.gz
    else
      echo "$weights_file"" exists"
    fi
  done
done
