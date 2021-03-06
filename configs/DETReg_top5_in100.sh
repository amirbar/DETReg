#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_top5_in100
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset imagenet100 --strategy topk --load_backbone swav --max_prop 5 --object_embedding_loss --lr_backbone 0 ${PY_ARGS}