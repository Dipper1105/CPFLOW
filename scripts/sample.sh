#!/usr/bin/env bash
# Generate treated morphology + transcriptome from control cells (stage-2 model).
set -e
export CUDA_VISIBLE_DEVICES=0

CKPT=${1:?"usage: sample.sh <checkpoint.pt>"}

python sample.py \
  --model "CPFlow-XL/2" \
  --ckpt "$CKPT" \
  --h5ad-path /data/pr/DiT_AIVCdiff/pr_tutorial/DiT_input_512_one_image_one_rna.h5ad \
  --image-column "merged_image" \
  --drug-column "compound" \
  --smiles-column "smiles" \
  --control-value "DMSO" \
  --source ctrl \
  --num-samples 500 \
  --batch-size 16 \
  --num-steps 50 \
  --method heun \
  --cfg-scale 1.5 \
  --image-size 512 \
  --sample-dir samples_cpflow
