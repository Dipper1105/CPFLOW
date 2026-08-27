#!/usr/bin/env bash
# Second image-focused fine-tuning pass.
#
# Starts from the first image-focused checkpoint and shifts the loss balance
# further toward image-latent velocity quality while keeping a small RNA term.
set -e
cd "$(dirname "$0")/.."

H5AD=/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad
CTRL=/data1/dataset/stem_cell/CPgenes/rna_ctrl_data_filtered.h5ad
IMGDIR=/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all
VAE=/data1/nicole/models/sd-vae-ft-ema
START_CKPT=/data1/nicole/CPgenes/CPFLOW/results_image_finetune/000-CPFlow-B-2-imageft/checkpoints/last.pt
RUN_DIR=/data1/nicole/CPgenes/CPFLOW/results_image_finetune_round2/000-CPFlow-B-2-imageft2
RESUME_CKPT="$RUN_DIR/checkpoints/start_from_imageft.pt"

mkdir -p "$RUN_DIR/checkpoints"
if [ ! -f "$RESUME_CKPT" ]; then
  cp "$START_CKPT" "$RESUME_CKPT"
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="/disk1/nicole/miniconda3/envs/MultiVCDiff/bin:$PATH"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} torchrun --nproc_per_node=1 --master_port=29543 train.py \
  --stage noise \
  --h5ad-path "$H5AD" \
  --image-dir "$IMGDIR" \
  --vae-path "$VAE" \
  --image-column merged_image \
  --drug-column compound \
  --smiles-column smiles \
  --image-size 512 \
  --model CPFlow-B/2 \
  --rna-tokens 8 \
  --fusion-every 4 \
  --flow-sigma 0.03 \
  --drug-dropout-prob 0.08 \
  --img-loss-weight 10.0 \
  --rna-loss-weight 0.05 \
  --align-weight 0.01 \
  --corr-weight 0.0 \
  --ctrl-rna-h5ad "$CTRL" \
  --global-batch-size 32 \
  --num-workers 8 \
  --lr 1e-5 \
  --epochs 8 \
  --log-every 20 \
  --ckpt-every 500 \
  --results-dir results_image_finetune_round2 \
  --resume-checkpoint "$RESUME_CKPT" \
  --wandb-run-name imageft2_from_imageft_img10_rna005
