#!/usr/bin/env bash
# Stage 2: control -> treated fine-tuning (CellFlux-v2 curriculum, step 2).
# Resume from the stage-1 checkpoint and learn the true perturbation displacement
# using same-batch control cells as the flow source.
#
# Set --batch-column to your plate/batch column so controls are paired within the
# same plate (batch-effect decoupling). If your h5ad has no batch column, drop the
# flag and controls are drawn from the global control pool.
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

STAGE1_CKPT=${1:?"usage: train_stage2_paired.sh <stage1_checkpoint.pt>"}

torchrun --nproc_per_node=4 --master_port=29511 train.py \
  --stage paired \
  --resume-checkpoint "$STAGE1_CKPT" \
  --h5ad-path /data/pr/DiT_AIVCdiff/pr_tutorial/DiT_input_512_one_image_one_rna.h5ad \
  --image-column "merged_image" \
  --drug-column "compound" \
  --smiles-column "smiles" \
  --control-value "DMSO" \
  --batch-column "plate" \
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
  --lr 5e-5 \
  --epochs 50 \
  --log-every 10 \
  --ckpt-every 5000 \
  --results-dir results_cpflow \
  --use-wandb \
  --wandb-project "CPFLOW" \
  --wandb-run-name "cpflow_stage2_paired"
