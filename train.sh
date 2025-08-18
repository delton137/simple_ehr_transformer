export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

TAG=aou_2022
DATA_DIR=/home/jupyter/workspaces/ehrtransformerbaseline/simple_ehr_transformer/processed_data_$TAG

python train.py \
    --tag $TAG \
    --data_dir $DATA_DIR \
