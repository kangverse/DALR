#!/bin/bash

# Run STS evaluation for a model trained on wiki+coco.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-1}"

OUT_DIR=result/mix_coco/best/${SEED}/mse

python src/evaluation.py \
      --model_name_or_path $OUT_DIR \
      --pooler cls_before_pooler \
      --task_set sts \
      --mode test
