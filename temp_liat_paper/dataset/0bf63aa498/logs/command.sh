/dccstor/alonhal1/anaconda3/envs/best-zero-shot-model-torch2.0/bin/python -m train --input_dir experiments/join_train_data/4b8e61c829 --train_df_name train.csv --dev_df_name dev.csv --learning_rate 5e-05 --max_seq_length 512 --base_model google/flan-t5-large --hypothesis_template custom_option_multi_class --num_epochs 3 --batch_size 8 --training_method t5 --version 1.0 --fp16 --output_dir experiments/train/0bf63aa498