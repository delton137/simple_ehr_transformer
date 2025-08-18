export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

TAG=aou_pre_2022
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/simple_ehr_transformer/processed_data_$TAG

# Train using settings from config.py
# All training parameters (batch_size, learning_rate, model architecture, etc.) 
# are now configured in config.py instead of command line arguments
python train.py \
    --tag $TAG \
    --data_dir $DATA_DIR
