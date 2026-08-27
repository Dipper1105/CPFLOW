#!/usr/bin/env bash
# Combined run: A + B + C all enabled at once.
#   --align-weight 0.1     : InfoNCE cross-modal contrastive alignment
#   --rna-tokens 8         : multi-token RNA representation
#   --ctrl-rna-h5ad + --stage noise : real DMSO controls as RNA flow source
#                                     (image source stays noise — no ctrl images)
#
# Checkpoint policy:
#   every 1000 steps -> write <step>.pt AND sync last.pt
#   end of each epoch -> refresh last.pt only
#   final -> refresh last.pt one more time
set -e
cd "$(dirname "$0")/.."

H5AD=/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad
CTRL=/data1/dataset/stem_cell/CPgenes/rna_ctrl_data_filtered.h5ad
IMGDIR=/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all
VAE=/data1/nicole/models/sd-vae-ft-ema

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="/disk1/nicole/miniconda3/envs/MultiVCDiff/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29530 train.py \
  --stage noise \
  --h5ad-path $H5AD \
  --image-dir $IMGDIR \
  --vae-path $VAE \
  --image-column merged_image \
  --drug-column compound \
  --smiles-column smiles \
  --image-size 512 \
  --model CPFlow-B/2 \
  --rna-tokens 8 \
  --fusion-every 4 \
  --flow-sigma 0.1 \
  --drug-dropout-prob 0.1 \
  --img-loss-weight 1.0 \
  --rna-loss-weight 1.0 \
  --align-weight 0.1 \
  --corr-weight 0.0 \
  --ctrl-rna-h5ad $CTRL \
  --global-batch-size 32 \
  --num-workers 8 \
  --lr 1e-4 \
  --epochs 60 \
  --log-every 20 \
  --ckpt-every 1000 \
  --results-dir results_combined \
  --wandb-run-name expCombined_A0.1_rnatok8_ctrlRNA
