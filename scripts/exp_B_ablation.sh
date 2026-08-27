#!/usr/bin/env bash
# Experiment B: does the stronger RNA conditioning representation help?
#
# Isolates the RNA tokenisation width (improvement B), the piece the plan calls
# out for fixing the 1-vs-T image/RNA token asymmetry:
#   baseline  : --rna-tokens 1   (single RNA token, MultiVCDiff-style)
#   treatment : --rna-tokens 8   (K latent RNA tokens through RNATokenizer)
# Everything else identical; align/corr weights fixed so ONLY rna-tokens differs.
#
# GPU 4 = baseline (1 token), GPU 5 = treatment (8 tokens). noise->treated stage.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

H5AD=/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad
IMGDIR=/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all
VAE=/data1/nicole/models/sd-vae-ft-ema
TORCHRUN=/disk1/nicole/miniconda3/envs/MultiVCDiff/bin/torchrun

COMMON="--stage noise \
  --h5ad-path $H5AD \
  --image-dir $IMGDIR \
  --vae-path $VAE \
  --image-column merged_image \
  --drug-column compound \
  --smiles-column smiles \
  --image-size 512 \
  --model CPFlow-B/2 \
  --fusion-every 4 \
  --flow-sigma 0.1 \
  --drug-dropout-prob 0.1 \
  --img-loss-weight 1.0 \
  --rna-loss-weight 1.0 \
  --align-weight 0.0 \
  --corr-weight 0.0 \
  --global-batch-size 32 \
  --num-workers 8 \
  --lr 1e-4 \
  --epochs 60 \
  --log-every 20 \
  --ckpt-every 4000 \
  --results-dir results_expB"

echo "Launching baseline (GPU 4, rna-tokens=1) and treatment (GPU 5, rna-tokens=8)..."

CUDA_VISIBLE_DEVICES=4 $TORCHRUN --nproc_per_node=1 --master_port=29522 train.py \
  $COMMON --rna-tokens 1 \
  --wandb-run-name expB_baseline_1token \
  > logs_expB_baseline.txt 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=5 $TORCHRUN --nproc_per_node=1 --master_port=29523 train.py \
  $COMMON --rna-tokens 8 \
  --wandb-run-name expB_8tokens \
  > logs_expB_treat.txt 2>&1 &
PID_TREAT=$!

echo "baseline PID=$PID_BASE (logs_expB_baseline.txt)"
echo "treat    PID=$PID_TREAT (logs_expB_treat.txt)"
wait $PID_BASE $PID_TREAT
echo "Experiment B finished."
