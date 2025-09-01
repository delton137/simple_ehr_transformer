TAG=aou_2022_2023_2024
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/omop_data_2021_2022_2023
 
python data_processor.py --data_path $DATA_DIR --tag $TAG -split_by_person --train_frac 0.7 --split_seed 42

TAG=aou_2024
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/omop_data_2024
 
python data_processor.py --data_path $DATA_DIR --tag $TAG --tokenization_spec processed_data_aou_2022/tokenization.yaml