DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/simple_ehr_transformer/processed_data_aou_10days

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

python train.py --tag aou_10days --data_dir $DATA_DIR --batch_size 8 --max_seq_len 512 --use_amp --num_workers 0

#torchrun --standalone --nproc_per_node=4 train.py \
#  --tag aou_10days \
#  --data_dir $DATA_DIR \
#  --batch_size 8 \
#  --grad_accum_steps 4 \
#  --max_seq_len 1024 \
#  --learning_rate 3e-4 \
#  --use_amp \
#  --num_workers 2