#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_2pct_coco
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --filter_pct 0.02 --dataset coco --pretrain exps/DETReg_top30_coco/checkpoint.pth --epochs 1000 --lr_drop 1000 ${PY_ARGS}