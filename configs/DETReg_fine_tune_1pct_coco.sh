#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_1pct_coco
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --filter_pct 0.01 --dataset coco --pretrain exps/DETReg_top30_in100/checkpoint.pth --epochs 150 ${PY_ARGS}