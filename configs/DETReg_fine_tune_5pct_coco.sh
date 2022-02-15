#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_5pct_coco
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --filter_pct 0.05 --dataset coco --pretrain exps/DETReg_top30_coco/checkpoint.pth --epochs 500 --lr_drop 500 ${PY_ARGS}