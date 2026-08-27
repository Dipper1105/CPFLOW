"""
CPFLOW training script (DDP), rectified-flow multimodal generator.

Two-stage curriculum from CellFlux v2:
  stage 1 ("noise")  : learn  noise -> (treated image, treated RNA).
                       Warms up the generator using all treated samples.
  stage 2 ("paired") : fine-tune  (control image/RNA) -> (treated image/RNA),
                       learning the true perturbation displacement.

Run stage 1, then resume its checkpoint with --stage paired.

Loss = w_img * flow_img + w_rna * flow_rna
       + w_align * InfoNCE(image, RNA projections)
       + w_corr  * gene-covariance regulariser
The image/RNA weights are actually applied here (MultiVCDiff parsed but ignored
them). Optional uncertainty weighting (Kendall 2018) balances the two modalities
automatically when --uncertainty-weighting is set.
"""
import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
import shutil
from time import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from diffusers.models import AutoencoderKL

from cpflow.models import CPFlow_models
from cpflow.flow import FlowTransport
from cpflow.multimodal_dataset import PairedCellDataset
from cpflow import losses


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in OrderedDict(model.named_parameters()).items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        return logging.getLogger(__name__)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger


def encode_image(vae, x):
    return vae.encode(x).latent_dist.sample().mul_(0.18215)


def main(args):
    assert torch.cuda.is_available(), "Training requires a GPU."
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    use_wandb = args.use_wandb and rank == 0
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"{args.model}-{args.stage}",
                   config=vars(args))

    # experiment dir
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if args.resume_checkpoint:
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_checkpoint))
        else:
            idx = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{args.results_dir}/{idx:03d}-{args.model.replace('/', '-')}-{args.stage}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None

    # data
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = PairedCellDataset(
        h5ad_path=args.h5ad_path,
        image_dir=args.image_dir,
        image_size=args.image_size,
        image_col=args.image_column,
        drug_col=args.drug_column,
        smiles_col=args.smiles_column,
        batch_col=args.batch_column,
        control_value=args.control_value,
        mode="paired" if args.stage == "paired" else "noise",
        transform=transform,
        normalize_rna=True,
        fp_size=args.fp_size,
        comb_num=args.comb_num,
        seed=args.global_seed,
        ctrl_rna_h5ad=args.ctrl_rna_h5ad,
    )

    # model
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    assert args.model in CPFlow_models, f"available: {list(CPFlow_models.keys())}"
    model = CPFlow_models[args.model](
        input_size=latent_size,
        in_channels=4,
        num_rna_features=args.num_rna_features,
        drug_fp_size=args.fp_size,
        rna_tokens=args.rna_tokens,
        fusion_every=args.fusion_every,
        drug_dropout_prob=args.drug_dropout_prob,
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    flow = FlowTransport(sigma=args.flow_sigma)
    vae_source = args.vae_path if args.vae_path else f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_source).to(device)
    requires_grad(vae, False)
    vae.eval()
    logger.info(f"CPFlow parameters: {sum(p.numel() for p in model.parameters()):,}")

    # optional learnable uncertainty weights (log-variance) for img/rna balancing
    log_vars = None
    params = list(model.parameters())
    if args.uncertainty_weighting:
        log_vars = nn.Parameter(torch.zeros(2, device=device))
        params = params + [log_vars]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)

    if args.resume_checkpoint:
        logger.info(f"Loading checkpoint: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        # optimizer state intentionally NOT restored across curriculum stages
        if args.resume_optimizer and "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for p in ema.parameters():
            dist.broadcast(p.data, src=0)

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                 rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.global_batch_size // dist.get_world_size(),
                        shuffle=False, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)

    update_ema(ema, model.module, decay=0)  # sync
    model.train()
    ema.eval()

    def _atomic_save(payload, dest):
        """torch.save to a tmp file then atomically rename onto dest.
        Prevents readers from seeing a half-written file if training is killed
        mid-write (which we hit in smoke-testing this policy)."""
        tmp = dest + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, dest)

    def _snapshot():
        return {"model": model.module.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "args": args,
                "step": train_steps, "epoch": epoch}

    log_steps = running = r_img = r_rna = r_align = r_corr = 0
    train_steps = 0
    epoch = 0
    start = time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            x_img1 = batch["image"].to(device)
            x_rna1 = batch["rna"].to(device)
            drug_fp = batch["drug_embedding"].to(device)

            with torch.no_grad():
                img1 = encode_image(vae, x_img1)
                if args.stage == "paired":
                    img0 = encode_image(vae, batch["image_ctrl"].to(device))
                    rna0 = batch["rna_ctrl"].to(device)
                else:  # noise source (image always noise here)
                    img0 = torch.randn_like(img1)
                    # experiment C: use real control RNA as the RNA flow source
                    # when provided; otherwise fall back to Gaussian noise.
                    if "rna_ctrl" in batch:
                        rna0 = batch["rna_ctrl"].to(device)
                    else:
                        rna0 = torch.randn_like(x_rna1)

            fb = flow.build_batch(img0, img1, rna0, x_rna1, device)
            img_v_pred, rna_v_pred, feats = model(
                fb["img_t"], fb["rna_t"], fb["t"], drug_fp, return_features=True
            )

            loss_img = losses.flow_matching_loss(img_v_pred, fb["img_v"]).mean()
            loss_rna = losses.flow_matching_loss(rna_v_pred, fb["rna_v"]).mean()
            loss_align = losses.info_nce_loss(feats["z_img"], feats["z_rna"], args.info_nce_temp)
            loss_corr = losses.rna_correlation_loss(fb["rna_t"], rna_v_pred, fb["rna_v"], fb["t"])

            if args.uncertainty_weighting:
                # Kendall et al. 2018:  1/(2 sigma^2) L + log sigma
                precision = torch.exp(-log_vars)
                loss_recon = (precision[0] * loss_img + 0.5 * log_vars[0]
                              + precision[1] * loss_rna + 0.5 * log_vars[1])
            else:
                loss_recon = args.img_loss_weight * loss_img + args.rna_loss_weight * loss_rna

            loss = loss_recon + args.align_weight * loss_align + args.corr_weight * loss_corr

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            update_ema(ema, model.module)

            running += loss.item()
            r_img += loss_img.item(); r_rna += loss_rna.item()
            r_align += loss_align.item(); r_corr += loss_corr.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                sps = log_steps / (time() - start)
                avg = torch.tensor(running / log_steps, device=device)
                dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                avg = avg.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) loss={avg:.4f} "
                    f"img={r_img/log_steps:.4f} rna={r_rna/log_steps:.4f} "
                    f"align={r_align/log_steps:.4f} corr={r_corr/log_steps:.4f} "
                    f"steps/s={sps:.2f}"
                )
                if use_wandb:
                    import wandb
                    wandb.log({"loss": avg, "img_loss": r_img/log_steps,
                               "rna_loss": r_rna/log_steps, "align_loss": r_align/log_steps,
                               "corr_loss": r_corr/log_steps, "steps_per_sec": sps,
                               "step": train_steps, "epoch": epoch})
                log_steps = running = r_img = r_rna = r_align = r_corr = 0
                start = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0 and rank == 0:
                # numbered ckpts are for evaluation: skip optimizer state
                # (~1.6 GB) so a 60-epoch run at 1k-step cadence stays ~500 GB
                # instead of ~1.3 TB. Full state (incl. opt) lives in last.pt.
                snap_eval = {k: v for k, v in _snapshot().items() if k != "opt"}
                path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                last = f"{checkpoint_dir}/last.pt"
                _atomic_save(snap_eval, path)
                _atomic_save(_snapshot(), last)   # last.pt keeps full state
                logger.info(f"Saved checkpoint to {path} (+ last.pt)")
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                dist.barrier()

        # end of epoch: refresh last.pt only (no numbered ckpt)
        if rank == 0:
            _atomic_save(_snapshot(), f"{checkpoint_dir}/last.pt")
            logger.info(f"[epoch {epoch}] refreshed last.pt @ step {train_steps}")
        dist.barrier()

    model.eval()
    # final flush: write last.pt one more time at end of training
    if rank == 0:
        _atomic_save(_snapshot(), f"{checkpoint_dir}/last.pt")
        logger.info(f"[final] flushed last.pt @ step {train_steps}")
    dist.barrier()
    if use_wandb:
        import wandb
        wandb.finish()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--h5ad-path", type=str, required=True)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--image-column", type=str, default="merged_image")
    p.add_argument("--drug-column", type=str, default="compound")
    p.add_argument("--smiles-column", type=str, default="smiles")
    p.add_argument("--batch-column", type=str, default=None,
                   help="plate/batch column for same-batch control pairing")
    p.add_argument("--control-value", type=str, default="DMSO")
    p.add_argument("--ctrl-rna-h5ad", type=str, default=None,
                   help="external control-RNA h5ad; used as the RNA flow source "
                        "in noise stage (experiment C, ctrl->treated on RNA)")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--comb-num", type=int, default=1)
    p.add_argument("--num-rna-features", type=int, default=977)
    # model
    p.add_argument("--model", type=str, default="CPFlow-XL/2")
    p.add_argument("--rna-tokens", type=int, default=8)
    p.add_argument("--fusion-every", type=int, default=4, help="0 disables cross-attn fusion")
    p.add_argument("--drug-dropout-prob", type=float, default=0.1)
    p.add_argument("--flow-sigma", type=float, default=0.1, help="noisy-interpolant std (0=plain RF)")
    # curriculum
    p.add_argument("--stage", type=str, choices=["noise", "paired"], default="noise")
    # loss weights
    p.add_argument("--img-loss-weight", type=float, default=1.0)
    p.add_argument("--rna-loss-weight", type=float, default=1.0)
    p.add_argument("--align-weight", type=float, default=0.1)
    p.add_argument("--corr-weight", type=float, default=0.05)
    p.add_argument("--info-nce-temp", type=float, default=0.1)
    p.add_argument("--uncertainty-weighting", action="store_true")
    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--global-batch-size", type=int, default=36)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    # vae / io
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    p.add_argument("--vae-path", type=str, default=None)
    p.add_argument("--results-dir", type=str, default="results_cpflow")
    p.add_argument("--resume-checkpoint", type=str, default=None)
    p.add_argument("--resume-optimizer", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=5000)
    # wandb
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="CPFLOW")
    p.add_argument("--wandb-run-name", type=str, default=None)
    args = p.parse_args()
    main(args)
