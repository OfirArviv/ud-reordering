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

for d in "$DIR"/*/     # list directories in the form "/tmp/dirname/"
do
    d=${d%*/}      # remove the trailing "/"
    echo "${d##*/}"    # print everything after the final "/"
    tar "–xvzf" "$d"/model.tar.gz
done
