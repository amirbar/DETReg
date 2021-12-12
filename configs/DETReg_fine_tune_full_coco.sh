#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_full_coco
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset coco --pretrain exps/DETReg_top30_in/checkpoint.pth ${PY_ARGS}