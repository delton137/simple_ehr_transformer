ROOT=/home/jupyter/workspaces/ehrtransformerbaselinecdr8

python build_year_split_rf.py \
  --tokenization_spec processed_data_aou_2021_2022/train/tokenization.yaml \
  --omop_2021 $ROOT/omop_data_2021_2022 \
  --omop_2022 $ROOT/omop_data_2023 \
  --target_concept_id 201826 \
  --out_dir processed_data_aou_2021_2022/year_split_rf