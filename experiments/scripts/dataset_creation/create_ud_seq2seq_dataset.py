import glob
import os.path
import conllu


# Code adapted from: https://github.com/bcmi220/seq2seq_parser/blob/master/data/scripts/make_dp_dataset.py
def conllu_to_seq2seq(ud_file_path: str, output_file_path: str):
    with open(ud_file_path, 'r', encoding='utf-8') as f, \
            open(output_file_path, 'x', encoding='utf-8') as out_f:
        for sent in conllu.parse_incr(f):
            sent_input_seq = " ".join([node['form'] for node in sent])
            sent_target_seq_arr = []
            for node in sent:
                if not isinstance(node['id'], int):
                    continue
                dep_ind = int(node['id'])
                head_ind = int(node['head'])
                if dep_ind > head_ind:
                    tag = 'L' + str(abs(dep_ind - head_ind))
                else:
                    tag = 'R' + str(abs(dep_ind - head_ind))
                dep_id = node['deprel']
                dep_id = dep_id.split(":")[0]
                tag = tag + " " + dep_id
                sent_target_seq_arr.append(tag)
            sent_target_seq = " ".join(sent_target_seq_arr)
            out_f.write(f'{sent_input_seq}\t{sent_target_seq}\n')


def create_seq2seq_dataset_script():
    conllu_dataset_root_dir = "experiments/processed_datasets/ud/conllu_format/"
    seq2seq_dataset_root_dir = 'experiments/processed_datasets/ud/seq2seq/'
    os.makedirs(seq2seq_dataset_root_dir, exist_ok=True)

    _dir, subdirs, files = list(os.walk(conllu_dataset_root_dir))[0]
    for subdir in subdirs:
        subdir_path = os.path.join(_dir, subdir)
        for f_path in glob.glob(f'{subdir_path}/*conllu'):
            basename = os.path.basename(f_path)
            output_subdir = os.path.join(seq2seq_dataset_root_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f'{basename}.tsv')
            if os.path.exists(output_path):
                print(f'{output_path} already exists! Skipping!')
                continue
            conllu_to_seq2seq(f_path, output_path)


if __name__ == "__main__":
    create_seq2seq_dataset_script()
