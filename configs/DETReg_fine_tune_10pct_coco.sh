#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_10pct_coco
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --filter_pct 0.1 --dataset coco --pretrain exps/DETReg_top30_in100/checkpoint0049.pth --epochs 150 ${PY_ARGS}