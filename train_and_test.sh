TAG=aou_pre2023_30000
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/simple_ehr_transformer/processed_data_$TAG

python train.py \
    --tag $TAG \
    --data_dir $DATA_DIR \
    --batch_size 8 \
    --max_seq_len 512 \
    --max_epochs 2

python test.py \
  --model_path models/aou_pre2023_30000/best_checkpoint.pth \
  --current_data_dir processed_data_aou_pre2023_30000 \
  --future_data_dir processed_data_aou_2023_30000 \
  --output_dir test_results_pre_to_2023 \
  --num_samples 10 \
  --max_gen_tokens 512\
  --temperature 1.0 \
  --top_k 50 \
  --top_p 0.9 \
  --patient_limit 300 \
  --targets CONDITION_201826,CONDITION_444070,CONDITION_316139


#ITION_201826 diabetes t2d   "CONDITION_444070",	#fatigue  "CONDITION_316139", #heart failure
  