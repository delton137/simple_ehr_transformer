python test.py \
  --debug_samples 1 \
  --model_path models/aou_pre2023_30000/latest_checkpoint.pth \
  --current_data_dir processed_data_aou_pre2023_30000 \
  --future_data_dir processed_data_aou_2023_30000 \
  --output_dir test_results_pre_to_2023 \
  --num_samples 18 \
  --max_gen_tokens 50000 \
  --temperature 1.0 \
  --top_k 50 \
  --patient_limit 100 \
  --targets CONDITION_201826,CONDITION_444070,CONDITION_316139 \
  --target_names "Type 2 Diabetes","Fatige","Heart Failure"


  