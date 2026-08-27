#!/usr/bin/env bash
# Stage 1: noise -> treated warmup (CellFlux-v2 curriculum, step 1).
# Trains the joint flow generator from Gaussian noise to treated cells.
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=29510 train.py \
  --stage noise \
  --h5ad-path /data/pr/DiT_AIVCdiff/pr_tutorial/DiT_input_512_one_image_one_rna.h5ad \
  --image-column "merged_image" \
  --drug-column "compound" \
  --smiles-column "smiles" \
  --image-size 512 \
  --model "CPFlow-XL/2" \
  --rna-tokens 8 \
  --fusion-every 4 \
  --flow-sigma 0.1 \
  --drug-dropout-prob 0.1 \
  --img-loss-weight 1.0 \
  --rna-loss-weight 1.0 \
  --align-weight 0.1 \
  --corr-weight 0.05 \
  --global-batch-size 36 \
  --num-workers 32 \
  --lr 1e-4 \
  --epochs 50 \
  --log-every 10 \
  --ckpt-every 5000 \
  --results-dir results_cpflow \
  --use-wandb \
  --wandb-project "CPFLOW" \
  --wandb-run-name "cpflow_stage1_noise"
