import argparse
import glob
import os


def convert_pointer_format_tsv_to_non_pointer_format(pointer_format_dataset_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    subdirs = list(os.walk(pointer_format_dataset_dir))[0][1]
    for d in subdirs:
        input_dir = f'{pointer_format_dataset_dir}/{d}'
        output_subdir = f'{output_dir}/{d}'
        os.makedirs(output_subdir, exist_ok=True)
        file_paths = glob.glob(f'{input_dir}/*.tsv')

        for file_path in file_paths:
            file_basename = os.path.basename(file_path)
            output_file_path = f'{output_subdir}/{file_basename}'
            with open(file_path, 'r', encoding='utf-8') as f, \
                    open(output_file_path, 'x', encoding='utf-8') as o_f:
                for line in f:
                    error = False
                    source_seq, target_seq = line.strip("\n").split("\t")
                    source_seq_arr = source_seq.split()
                    non_pointer_target_seq_arr = []
                    for tok in target_seq.split():
                        if tok.startswith("@ptr"):
                            pointer_idx = int(tok.split("@ptr")[1])
                            try:
                                non_pointer_target_seq_arr.append(source_seq_arr[pointer_idx])
                            except:
                                error = True
                        else:
                            non_pointer_target_seq_arr.append(tok)
                    if error:
                        continue
                    non_pointer_target_seq = " ".join(non_pointer_target_seq_arr)
                    o_f.write(f'{source_seq}\t{non_pointer_target_seq}\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-i", "--input-dir", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)

    args = argparser.parse_args()

    convert_pointer_format_tsv_to_non_pointer_format(args.input_dir, args.output_dir)

