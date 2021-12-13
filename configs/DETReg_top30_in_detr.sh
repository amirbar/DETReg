#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_top30_in_detr
PY_ARGS=${@:1}

python -u main.py --model detr --output_dir ${EXP_DIR} --dataset imagenet --strategy topk --load_backbone swav --max_prop 30 --object_embedding_loss --object_embedding_coef 1 --lr_backbone 0 --pre_norm --epochs 60 ${PY_ARGS}
