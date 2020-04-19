# Computes the estimates for corpora and dumps them into JSON files

import json
import importlib
import pathlib

import UDLib as U
import Estimation as E
import argparse


def conllu2trees(path):
    with open(path, 'r', encoding='utf-8') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [U.UDTree(*U.conll2graph(block)) for block in blocks]


def dump_estimates(est_dict, path):
    est_dict_str_keys = {}
    for deprel in est_dict:
        est_dict_str_keys[deprel] = {
            f'{r1}->{r2}': count for (r1, r2), count in est_dict[deprel].items()
        }
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(est_dict_str_keys, out, indent=2)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-i", "--input-file", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)

    args = argparser.parse_args()

    trees = conllu2trees(args.input_file)
    estimates = E.get_ml_directionality_estimates(trees)
    file_name = args.input_file.split("/")[-1]
    dump_estimates(estimates, f'{args.output_dir}/{file_name}.estimates.json')
