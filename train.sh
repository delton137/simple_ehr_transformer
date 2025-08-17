TAG=aou_pre2023
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/simple_ehr_transformer/processed_data_$TAG

python train.py --tag $TAG --data_dir $DATA_DIR --count_tokens
