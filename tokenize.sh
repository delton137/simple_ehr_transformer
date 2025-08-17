TAG=aou_2022
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/omop_data_2022
 
python data_processor.py --data_path $DATA_DIR --tag $TAG 



TAG=aou_2023
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/omop_data_2023
 
python data_processor.py --data_path $DATA_DIR --tag $TAG --tokenization_spec processed_data_aou_2022/tokenization.yaml