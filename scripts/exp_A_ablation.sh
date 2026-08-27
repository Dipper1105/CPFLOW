#!/usr/bin/env bash
# Experiment A: does cross-modal contrastive alignment (InfoNCE) help?
#
# Clean ablation on the treated set (each cell has paired image+RNA):
#   baseline  : --align-weight 0    (no contrastive term)
#   treatment : --align-weight 0.1  (InfoNCE image<->RNA)
# Everything else identical. corr-weight=0 so ONLY the InfoNCE term differs.
#
# Two single-GPU runs launched in parallel (GPU 2 = baseline, GPU 3 = align).
# noise->treated stage, since controls in this dataset are RNA-only (no images).
set -e
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/disk1/nicole/miniconda3/envs/MultiVCDiff/bin/python
H5AD=/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad
IMGDIR=/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all
VAE=/data1/nicole/models/sd-vae-ft-ema

COMMON="--stage noise \
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
  --corr-weight 0.0 \
  --global-batch-size 32 \
  --num-workers 8 \
  --lr 1e-4 \
  --epochs 60 \
  --log-every 20 \
  --ckpt-every 4000 \
  --results-dir results_expA"

echo "Launching baseline (GPU 2, align=0) and treatment (GPU 3, align=0.1)..."

CUDA_VISIBLE_DEVICES=2 /disk1/nicole/miniconda3/envs/MultiVCDiff/bin/torchrun --nproc_per_node=1 --master_port=29520 train.py \
  $COMMON --align-weight 0.0 \
  --wandb-run-name expA_baseline_noalign \
  > logs_expA_baseline.txt 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=3 /disk1/nicole/miniconda3/envs/MultiVCDiff/bin/torchrun --nproc_per_node=1 --master_port=29521 train.py \
  $COMMON --align-weight 0.1 \
  --wandb-run-name expA_align0.1 \
  > logs_expA_align.txt 2>&1 &
PID_ALIGN=$!

echo "baseline PID=$PID_BASE (logs_expA_baseline.txt)"
echo "align    PID=$PID_ALIGN (logs_expA_align.txt)"
wait $PID_BASE $PID_ALIGN
echo "Both runs finished."
