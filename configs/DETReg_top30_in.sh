#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_top30_in
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset imagenet --strategy topk --load_backbone swav --max_prop 30 --object_embedding_loss --lr_backbone 0 --epochs 5 ${PY_ARGS}