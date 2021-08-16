#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_10pct_pascal
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset_file voc --filter_pct 0.1 --dataset voc --eval_every 20 --pretrain exps/DETReg_top30_in100/checkpoint.pth ${PY_ARGS}