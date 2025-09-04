MODEL_ROOT=/home/jupyter/workspaces/ehrtransformerbaselinecdr8/simple_ehr_transformer/models/aou_2021_2022

python test.py \
  --debug_samples 1 \
  --model_path $MODEL_ROOT/latest_checkpoint.pth \
  --current_data_dir processed_data_aou_2021_2022 \
  --future_data_dir processed_data_aou_2023 \
  --output_dir test_results_aou_2021_2022 \
  --num_samples 18 \
  --max_gen_tokens 50000 \
  --temperature 1.0 \
  --top_k 50 \
  --patient_limit 100 \
  --targets CONDITION_201826,CONDITION_444070,CONDITION_316139 \
  --target_names "Type 2 Diabetes","Fatige","Heart Failure"


  