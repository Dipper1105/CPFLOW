#!/usr/bin/env bash
# Experiment C: does the rectified-flow ctrl->treated paradigm help?
#
# This corpus has NO control images (controls in rna_ctrl_data_filtered.h5ad are
# RNA-only), so the data-supported form of C runs the flow with the RNA source =
# real DMSO control and the image source = noise. That isolates ctrl->treated on
# the modality where controls actually exist.
#   baseline  : RNA source = Gaussian noise        (noise->treated)
#   treatment : RNA source = real DMSO control RNA  (ctrl->treated on RNA)
# Both use rectified flow; the ONLY difference is the RNA source distribution.
#
# GPU 6 = baseline (noise RNA source), GPU 7 = treatment (real control RNA).
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

H5AD=/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad
IMGDIR=/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all
CTRL_RNA=/data1/dataset/stem_cell/CPgenes/rna_ctrl_data_filtered.h5ad
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
  --rna-tokens 8 \
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
  --results-dir results_expC"

echo "Launching baseline (GPU 6, noise RNA source) and treatment (GPU 7, real control RNA)..."

CUDA_VISIBLE_DEVICES=6 $TORCHRUN --nproc_per_node=1 --master_port=29524 train.py \
  $COMMON \
  --wandb-run-name expC_baseline_noiseRNA \
  > logs_expC_baseline.txt 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=7 $TORCHRUN --nproc_per_node=1 --master_port=29525 train.py \
  $COMMON --ctrl-rna-h5ad $CTRL_RNA \
  --wandb-run-name expC_ctrlRNA \
  > logs_expC_treat.txt 2>&1 &
PID_TREAT=$!

echo "baseline PID=$PID_BASE (logs_expC_baseline.txt)"
echo "treat    PID=$PID_TREAT (logs_expC_treat.txt)"
wait $PID_BASE $PID_TREAT
echo "Experiment C finished."
