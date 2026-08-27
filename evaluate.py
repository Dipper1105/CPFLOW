"""
Evaluate a CPFLOW checkpoint on the treated set.

Two metric families, both cheap and checkpoint-comparable:

1. Cross-modal retrieval (measures representation alignment — the point of exp A).
   Encode held-out image+RNA pairs through the model's projection heads (a clean
   t=0 forward, no noise), then compute image<->RNA retrieval Recall@k and the
   mean cosine of matched pairs. Higher = better aligned.

2. RNA generation fidelity (measures exp B / exp C).
   For each drug, integrate the ODE from the appropriate source to produce RNA,
   then compare to the real per-drug mean profile via Pearson r and MSE.

Usage:
  python evaluate.py --ckpt <ckpt.pt> --rna-tokens 8 [--source noise|ctrl-rna]
  # for exp C treatment pass --ctrl-rna-h5ad and --source ctrl-rna
"""
import argparse
import numpy as np
import torch
import scanpy as sc
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL

from cpflow.models import CPFlow_models
from cpflow.flow import FlowTransport
from cpflow.multimodal_dataset import drug_encoder


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


@torch.no_grad()
def retrieval_metrics(model, vae, adata, obs, args, device, n=256):
    """Recall@1/5 and matched-pair cosine from the projection heads at t≈0."""
    idx = np.random.RandomState(0).choice(adata.n_obs, size=min(n, adata.n_obs), replace=False)
    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    X = dense(adata.X)
    rmin, rmax = X.min(0), X.max(0)
    rng = rmax - rmin; rng[rng == 0] = 1.0

    zi_all, zr_all = [], []
    for s in range(0, len(idx), args.batch_size):
        chunk = idx[s:s + args.batch_size]
        imgs, rnas = [], []
        for i in chunk:
            xi = 2 * ((dense(adata.X[int(i)]).flatten() - rmin) / rng) - 1
            rnas.append(torch.from_numpy(xi).float())
            p = obs[args.image_column].iloc[int(i)]
            if args.image_dir:
                import os; p = os.path.join(args.image_dir, p)
            imgs.append(tfm(Image.open(p).convert("RGB")))
        img = torch.stack(imgs).to(device)
        rna = torch.stack(rnas).to(device)
        img_lat = vae.encode(img).latent_dist.sample().mul_(0.18215)
        fp = torch.zeros(len(chunk), args.fp_size, device=device)  # cond irrelevant to proj
        t0 = torch.full((len(chunk),), 1e-3, device=device)
        _, _, feats = model(img_lat, rna, t0, fp, return_features=True)
        zi_all.append(feats["z_img"]); zr_all.append(feats["z_rna"])
    zi = torch.cat(zi_all); zr = torch.cat(zr_all)
    sim = zi @ zr.t()
    m = zi.shape[0]
    labels = torch.arange(m, device=device)
    rank = sim.argsort(dim=1, descending=True)
    r1 = (rank[:, 0] == labels).float().mean().item()
    r5 = (rank[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    matched_cos = sim.diag().mean().item()
    return {"recall@1": r1, "recall@5": r5, "matched_cosine": matched_cos, "n": m}


@torch.no_grad()
def rna_fidelity(model, flow, adata, obs, args, device, ctrl_rna=None):
    """Per-drug Pearson r and MSE between generated and real mean RNA profile."""
    X = dense(adata.X)
    rmin, rmax = X.min(0), X.max(0)
    rng = rmax - rmin; rng[rng == 0] = 1.0
    drug_to_smiles = dict(zip(obs[args.drug_column], obs[args.smiles_column]))
    latent = args.image_size // 8

    rs, mses = [], []
    for drug in sorted(obs[args.drug_column].unique().tolist()):
        smi = drug_to_smiles.get(drug)
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        real = X[(obs[args.drug_column] == drug).values].mean(0)
        fp = torch.tensor(drug_encoder([smi], num_bits=args.fp_size), dtype=torch.float32, device=device)
        bs = args.gen_per_drug
        img0 = torch.randn(bs, 4, latent, latent, device=device)
        if ctrl_rna is not None:
            j = np.random.choice(ctrl_rna.shape[0], size=bs)
            rna0 = ctrl_rna[j].to(device)
        else:
            rna0 = torch.randn(bs, args.num_rna_features, device=device)
        _, rna1 = flow.sample_ode(model, img0, rna0, fp.expand(bs, -1),
                                  num_steps=args.num_steps, cfg_scale=args.cfg_scale, method="heun")
        gen = rna1.cpu().numpy()
        gen_orig = ((gen + 1) / 2) * rng + rmin
        gen_mean = gen_orig.mean(0)
        rs.append(np.corrcoef(gen_mean, real)[0, 1])
        mses.append(float(((gen_mean - real) ** 2).mean()))
    return {"rna_pearson_mean": float(np.mean(rs)), "rna_mse_mean": float(np.mean(mses)),
            "n_drugs": len(rs)}


def main(args):
    torch.set_grad_enabled(False)
    device = 0
    torch.cuda.set_device(device)

    adata = sc.read_h5ad(args.h5ad_path)
    obs = adata.obs
    model = CPFlow_models[args.model](
        input_size=args.image_size // 8, in_channels=4,
        num_rna_features=args.num_rna_features, drug_fp_size=args.fp_size,
        rna_tokens=args.rna_tokens, fusion_every=args.fusion_every,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=f"cuda:{device}")
    state = ckpt["ema"] if (args.use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(state); model.eval()

    flow = FlowTransport(sigma=0.0)
    vae = AutoencoderKL.from_pretrained(args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    ctrl_rna = None
    if args.ctrl_rna_h5ad:
        cad = sc.read_h5ad(args.ctrl_rna_h5ad)
        Xc = dense(cad.X).astype(np.float32)
        X = dense(adata.X); rmin, rmax = X.min(0), X.max(0); rng = rmax - rmin; rng[rng == 0] = 1.0
        ctrl_rna = torch.from_numpy(2 * ((Xc - rmin) / rng) - 1).float()

    print(f"# checkpoint: {args.ckpt}")
    ret = retrieval_metrics(model, vae, adata, obs, args, device, n=args.retrieval_n)
    print("retrieval:", ret)
    fid = rna_fidelity(model, flow, adata, obs, args, device,
                       ctrl_rna=ctrl_rna if args.source == "ctrl-rna" else None)
    print("rna_fidelity:", fid)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default="CPFlow-B/2")
    p.add_argument("--h5ad-path", type=str,
                   default="/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad")
    p.add_argument("--image-dir", type=str,
                   default="/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all")
    p.add_argument("--image-column", type=str, default="merged_image")
    p.add_argument("--drug-column", type=str, default="compound")
    p.add_argument("--smiles-column", type=str, default="smiles")
    p.add_argument("--ctrl-rna-h5ad", type=str, default=None)
    p.add_argument("--source", type=str, choices=["noise", "ctrl-rna"], default="noise")
    p.add_argument("--rna-tokens", type=int, default=8)
    p.add_argument("--fusion-every", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-rna-features", type=int, default=977)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--retrieval-n", type=int, default=256)
    p.add_argument("--gen-per-drug", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--vae", type=str, default="ema")
    p.add_argument("--vae-path", type=str, default="/data1/nicole/models/sd-vae-ft-ema")
    p.add_argument("--use-ema", action="store_true", default=True)
    args = p.parse_args()
    main(args)
