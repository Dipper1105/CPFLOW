"""
CPFLOW inference: generate treated cell morphology + transcriptome for each drug
by integrating the learned velocity field with an ODE solver.

Two source modes matching the two training stages:
  --source noise : start from Gaussian noise (use a stage-1 checkpoint).
  --source ctrl  : start from real control (DMSO) cells drawn from the h5ad,
                   i.e. the control->treated transport (use a stage-2 checkpoint).

Outputs, per drug: PNG images and per-sample RNA CSVs (denormalised back to the
original L1000 scale), mirroring MultiVCDiff/sample_ddp.py's layout.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from cpflow.models import CPFlow_models
from cpflow.flow import FlowTransport
from cpflow.multimodal_dataset import drug_encoder


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    device = 0
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    adata = sc.read_h5ad(args.h5ad_path)
    X = dense(adata.X)
    rna_min, rna_max = X.min(0), X.max(0)
    rna_range = rna_max - rna_min
    rna_range[rna_range == 0] = 1.0

    latent_size = args.image_size // 8
    model = CPFlow_models[args.model](
        input_size=latent_size,
        in_channels=4,
        num_rna_features=args.num_rna_features,
        drug_fp_size=args.fp_size,
        rna_tokens=args.rna_tokens,
        fusion_every=args.fusion_every,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=f"cuda:{device}")
    state = ckpt["ema"] if (args.use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    flow = FlowTransport(sigma=0.0)  # deterministic transport at inference
    vae_source = args.vae_path if args.vae_path else f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_source).to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    obs = adata.obs
    drug_to_smiles = dict(zip(obs[args.drug_column], obs[args.smiles_column]))
    drug_names = sorted(obs[args.drug_column].unique().tolist())
    if args.drugs:
        drug_names = [d for d in drug_names if d in set(args.drugs)]

    # control-row indices for the ctrl source mode
    ctrl_rows = np.where(obs[args.drug_column].astype(str) == args.control_value)[0]

    def load_ctrl_batch(n):
        idx = np.random.choice(ctrl_rows, size=n, replace=len(ctrl_rows) < n)
        imgs, rnas = [], []
        for i in idx:
            xi = 2 * ((dense(adata.X[i]).flatten() - rna_min) / rna_range) - 1
            rnas.append(torch.from_numpy(xi).float())
            path = obs[args.image_column].iloc[int(i)]
            if args.image_dir:
                path = os.path.join(args.image_dir, path)
            imgs.append(transform(Image.open(path).convert("RGB")))
        img = torch.stack(imgs).to(device)
        rna0 = torch.stack(rnas).to(device)
        img0 = vae.encode(img).latent_dist.sample().mul_(0.18215)
        return img0, rna0

    for drug in drug_names:
        smiles = drug_to_smiles.get(drug)
        if smiles is None or pd.isna(smiles):
            print(f"skip {drug}: no SMILES")
            continue
        fp = drug_encoder([smiles], num_bits=args.fp_size)
        fp = torch.tensor(fp, dtype=torch.float32, device=device)

        out_dir = os.path.join(args.sample_dir, args.model.replace("/", "-"), str(drug))
        rna_dir = os.path.join(out_dir, "rna")
        os.makedirs(rna_dir, exist_ok=True)

        generated = 0
        n = args.batch_size
        iters = (args.num_samples + n - 1) // n
        for _ in tqdm(range(iters), desc=str(drug)):
            bs = min(n, args.num_samples - generated)
            if bs == 0:
                break
            if args.source == "ctrl":
                img0, rna0 = load_ctrl_batch(bs)
            else:
                img0 = torch.randn(bs, 4, latent_size, latent_size, device=device)
                rna0 = torch.randn(bs, args.num_rna_features, device=device)

            drug_fp = fp.expand(bs, -1)
            img1, rna1 = flow.sample_ode(
                model, img0, rna0, drug_fp,
                num_steps=args.num_steps, cfg_scale=args.cfg_scale, method=args.method,
            )
            images = vae.decode(img1 / 0.18215).sample
            rna_np = rna1.cpu().numpy()
            rna_orig = ((rna_np + 1) / 2) * rna_range + rna_min

            for j in range(bs):
                save_image(images[j], f"{out_dir}/{generated:06d}.png",
                           normalize=True, value_range=(-1, 1))
                np.savetxt(f"{rna_dir}/{generated:06d}.csv", rna_orig[j], delimiter=",")
                generated += 1

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="CPFlow-XL/2", choices=list(CPFlow_models.keys()))
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--h5ad-path", type=str, required=True)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--image-column", type=str, default="merged_image")
    p.add_argument("--drug-column", type=str, default="compound")
    p.add_argument("--smiles-column", type=str, default="smiles")
    p.add_argument("--control-value", type=str, default="DMSO")
    p.add_argument("--source", type=str, choices=["noise", "ctrl"], default="ctrl")
    p.add_argument("--drugs", type=str, nargs="*", default=None, help="subset of drugs to generate")
    p.add_argument("--sample-dir", type=str, default="samples_cpflow")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--method", type=str, choices=["euler", "heun"], default="heun")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    p.add_argument("--num-rna-features", type=int, default=977)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--rna-tokens", type=int, default=8)
    p.add_argument("--fusion-every", type=int, default=4)
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    p.add_argument("--vae-path", type=str, default=None)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
