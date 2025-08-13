TAG=aou_pre2023_30000
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/simple_ehr_transformer/processed_data_$TAG

python train.py --tag $TAG --data_dir $DATA_DIR --batch_size 8 --max_seq_len 512  --num_workers 0
