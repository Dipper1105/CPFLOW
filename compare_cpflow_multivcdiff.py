"""
Unified RNA-generation evaluation for CPFLOW combined vs original MultiVCDiff.

The script uses one treated h5ad, one RNA min/max inverse transform, and the
same per-drug Pearson/MSE metrics for both models. It also exports publication
style comparison figures and drug-level showcase panels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
MULTIVCDIFF_ROOT = REPO_ROOT / "MultiVCDiff"

sys.path.insert(0, str(ROOT))
from cpflow.flow import FlowTransport  # noqa: E402
from cpflow.models import CPFlow_models  # noqa: E402
from cpflow.multimodal_dataset import drug_encoder as cpflow_drug_encoder  # noqa: E402


@dataclass
class EvalBundle:
    adata: sc.AnnData
    obs: pd.DataFrame
    x: np.ndarray
    rmin: np.ndarray
    rmax: np.ndarray
    rng: np.ndarray
    drug_names: List[str]
    drug_to_smiles: Dict[str, str]
    real_means: Dict[str, np.ndarray]
    global_mean: np.ndarray
    ctrl_rna_norm: Optional[torch.Tensor]
    gene_names: List[str]


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bundle(args) -> EvalBundle:
    adata = sc.read_h5ad(args.h5ad_path)
    obs = adata.obs.copy()
    x = dense(adata.X).astype(np.float32)
    rmin = x.min(axis=0)
    rmax = x.max(axis=0)
    rng = rmax - rmin
    rng[rng == 0] = 1.0

    drug_names = sorted(obs[args.drug_column].unique().tolist())
    if args.max_drugs is not None:
        drug_names = drug_names[: args.max_drugs]
    drug_to_smiles = {}
    real_means = {}
    for drug in drug_names:
        mask = (obs[args.drug_column] == drug).values
        smi = obs.loc[mask, args.smiles_column].dropna()
        if len(smi) > 0:
            drug_to_smiles[drug] = str(smi.iloc[0])
        real_means[drug] = x[mask].mean(axis=0)

    ctrl_rna_norm = None
    if args.ctrl_rna_h5ad:
        ctrl = sc.read_h5ad(args.ctrl_rna_h5ad)
        xc = dense(ctrl.X).astype(np.float32)
        ctrl_rna_norm = torch.from_numpy(2 * ((xc - rmin) / rng) - 1).float()

    gene_names = [str(g) for g in adata.var_names]
    return EvalBundle(
        adata=adata,
        obs=obs,
        x=x,
        rmin=rmin,
        rmax=rmax,
        rng=rng,
        drug_names=drug_names,
        drug_to_smiles=drug_to_smiles,
        real_means=real_means,
        global_mean=x.mean(axis=0),
        ctrl_rna_norm=ctrl_rna_norm,
        gene_names=gene_names,
    )


def rows_to_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inverse_rna(norm_rna: np.ndarray, bundle: EvalBundle) -> np.ndarray:
    return ((norm_rna + 1.0) / 2.0) * bundle.rng + bundle.rmin


def metric_rows(
    model_name: str,
    pred_means: Dict[str, np.ndarray],
    bundle: EvalBundle,
    sampling_steps: int,
    gen_per_drug: int,
) -> List[Dict]:
    rows = []
    for drug, pred in pred_means.items():
        real = bundle.real_means[drug]
        rows.append(
            {
                "model": model_name,
                "drug": drug,
                "sampling_steps": sampling_steps,
                "gen_per_drug": gen_per_drug,
                "pearson": pearson(pred, real),
                "mse": float(((pred - real) ** 2).mean()),
                "mae": float(np.abs(pred - real).mean()),
            }
        )
    return rows


def save_pred_matrix(pred_means: Dict[str, np.ndarray], bundle: EvalBundle, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(pred_means, orient="index", columns=bundle.gene_names)
    df.index.name = "drug"
    df.to_csv(path)


@torch.no_grad()
def eval_cpflow(args, bundle: EvalBundle, device: torch.device) -> Dict[str, np.ndarray]:
    model = CPFlow_models[args.cpflow_model](
        input_size=args.image_size // 8,
        in_channels=4,
        num_rna_features=args.num_rna_features,
        drug_fp_size=args.fp_size,
        rna_tokens=args.cpflow_rna_tokens,
        fusion_every=args.cpflow_fusion_every,
    ).to(device)
    ckpt = torch.load(args.cpflow_ckpt, map_location="cpu")
    state = ckpt["ema"] if args.use_ema and "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    del ckpt, state
    gc.collect()

    flow = FlowTransport(sigma=0.0)
    latent = args.image_size // 8
    pred_means: Dict[str, np.ndarray] = {}

    for idx, drug in enumerate(bundle.drug_names, start=1):
        smi = bundle.drug_to_smiles.get(drug)
        if smi is None:
            continue
        set_seed(args.seed + idx)
        fp = torch.tensor(cpflow_drug_encoder([smi], num_bits=args.fp_size), dtype=torch.float32, device=device)
        samples = []
        remaining = args.cpflow_gen_per_drug
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            img0 = torch.randn(bs, 4, latent, latent, device=device)
            if bundle.ctrl_rna_norm is not None and args.cpflow_source == "ctrl-rna":
                choice = np.random.choice(bundle.ctrl_rna_norm.shape[0], size=bs)
                rna0 = bundle.ctrl_rna_norm[choice].to(device)
            else:
                rna0 = torch.randn(bs, args.num_rna_features, device=device)
            _, rna1 = flow.sample_ode(
                model,
                img0,
                rna0,
                fp.expand(bs, -1),
                num_steps=args.cpflow_num_steps,
                cfg_scale=args.cpflow_cfg_scale,
                method=args.cpflow_method,
            )
            samples.append(rna1.cpu().numpy())
            remaining -= bs
        gen = inverse_rna(np.concatenate(samples, axis=0), bundle)
        pred_means[drug] = gen.mean(axis=0)
        print(f"[CPFLOW] {idx:02d}/{len(bundle.drug_names)} {drug}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return pred_means


def import_multivcdiff_modules():
    sys.path.insert(0, str(MULTIVCDIFF_ROOT))
    mvc_models = importlib.import_module("models")
    diffusion = importlib.import_module("diffusion")
    mvc_dataset = importlib.import_module("multimodal_dataset")
    return mvc_models, diffusion, mvc_dataset


@torch.no_grad()
def eval_multivcdiff(args, bundle: EvalBundle, device: torch.device) -> Dict[str, np.ndarray]:
    mvc_models, diffusion_mod, mvc_dataset = import_multivcdiff_modules()
    model = mvc_models.DiTMultimodal_models[args.mvc_model](
        input_size=args.image_size // 8,
        in_channels=4,
        num_drug_classes=args.mvc_num_drug_classes,
        num_rna_features=args.num_rna_features,
        drug_fp_size=args.fp_size,
    ).to(device)
    ckpt = torch.load(args.mvc_ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    del ckpt, state
    gc.collect()

    diffusion = diffusion_mod.create_diffusion(str(args.mvc_num_steps))
    latent = args.image_size // 8
    pred_means: Dict[str, np.ndarray] = {}

    for idx, drug in enumerate(bundle.drug_names, start=1):
        smi = bundle.drug_to_smiles.get(drug)
        if smi is None:
            continue
        set_seed(args.seed + 1000 + idx)
        fp_np = mvc_dataset.Drug_encoder([smi], num_Bits=args.fp_size)
        fp = torch.tensor(fp_np, dtype=torch.float32, device=device)
        samples = []
        remaining = args.mvc_gen_per_drug
        while remaining > 0:
            bs = min(args.mvc_batch_size, remaining)
            img = torch.randn(bs, 4, latent, latent, device=device)
            rna = torch.randn(bs, args.num_rna_features, device=device)
            model_kwargs = {"drug_fp": fp.expand(bs, -1), "rna": rna}
            final_sample = None
            for sample in diffusion.p_sample_loop_progressive(
                model=model,
                img=img,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
            ):
                final_sample = sample
            if final_sample is None:
                raise RuntimeError(f"No MultiVCDiff samples for {drug}")
            samples.append(final_sample["rna_sample"].cpu().numpy())
            remaining -= bs
        gen = inverse_rna(np.concatenate(samples, axis=0), bundle)
        pred_means[drug] = gen.mean(axis=0)
        print(f"[MultiVCDiff] {idx:02d}/{len(bundle.drug_names)} {drug}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return pred_means


def summarize(per_drug: pd.DataFrame, output_dir: Path, extra_rows: Optional[List[Dict]] = None) -> pd.DataFrame:
    rows = []
    for model, df in per_drug.groupby("model"):
        rows.append(
            {
                "model": model,
                "n_drugs": int(df.shape[0]),
                "sampling_steps": int(df["sampling_steps"].iloc[0]),
                "gen_per_drug": int(df["gen_per_drug"].iloc[0]),
                "pearson_mean": float(df["pearson"].mean()),
                "pearson_median": float(df["pearson"].median()),
                "mse_mean": float(df["mse"].mean()),
                "mse_median": float(df["mse"].median()),
                "mae_mean": float(df["mae"].mean()),
                "mae_median": float(df["mae"].median()),
            }
        )
    if extra_rows:
        rows.extend(extra_rows)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    return summary


def merge_deltas(per_drug: pd.DataFrame) -> pd.DataFrame:
    wide = per_drug.pivot(index="drug", columns="model", values=["pearson", "mse", "mae"])
    flat = pd.DataFrame(index=wide.index)
    for metric in ["pearson", "mse", "mae"]:
        for model in per_drug["model"].unique():
            flat[f"{metric}_{model}"] = wide[(metric, model)]
    if "CPFLOW_combined" in per_drug["model"].unique() and "MultiVCDiff_original" in per_drug["model"].unique():
        flat["delta_pearson_cpflow_minus_multivcdiff"] = (
            flat["pearson_CPFLOW_combined"] - flat["pearson_MultiVCDiff_original"]
        )
        flat["delta_mse_cpflow_minus_multivcdiff"] = (
            flat["mse_CPFLOW_combined"] - flat["mse_MultiVCDiff_original"]
        )
    return flat.reset_index()


def save_pub(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_outputs(
    per_drug: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    bundle: EvalBundle,
    pred_paths: Dict[str, Path],
    output_dir: Path,
    showcase_n: int,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    palette = {"CPFLOW_combined": "#3B7EA1", "MultiVCDiff_original": "#B45F4D"}

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    sns.barplot(data=per_drug, x="model", y="pearson", palette=palette, errorbar=("ci", 95), ax=ax1)
    ax1.set_xlabel("")
    ax1.set_ylabel("Per-drug RNA Pearson")
    ax1.tick_params(axis="x", rotation=25)

    sns.barplot(data=per_drug, x="model", y="mse", palette=palette, errorbar=("ci", 95), ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("Per-drug RNA MSE")
    ax2.tick_params(axis="x", rotation=25)

    if {"pearson_CPFLOW_combined", "pearson_MultiVCDiff_original"}.issubset(deltas.columns):
        ax3.scatter(
            deltas["pearson_MultiVCDiff_original"],
            deltas["pearson_CPFLOW_combined"],
            s=18,
            color="#555555",
            alpha=0.75,
        )
        lo = min(ax3.get_xlim()[0], ax3.get_ylim()[0])
        hi = max(ax3.get_xlim()[1], ax3.get_ylim()[1])
        ax3.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, ls="--")
        ax3.set_xlim(lo, hi)
        ax3.set_ylim(lo, hi)
        ax3.set_xlabel("MultiVCDiff Pearson")
        ax3.set_ylabel("CPFLOW Pearson")
        ax3.set_title("Per-drug paired comparison", fontsize=8)

    if {"delta_pearson_cpflow_minus_multivcdiff", "delta_mse_cpflow_minus_multivcdiff"}.issubset(deltas.columns):
        plot_df = deltas.sort_values("delta_pearson_cpflow_minus_multivcdiff", ascending=False)
        sns.barplot(
            data=plot_df,
            x="drug",
            y="delta_pearson_cpflow_minus_multivcdiff",
            color="#3B7EA1",
            ax=ax4,
        )
        ax4.axhline(0, color="#333333", lw=0.8)
        ax4.set_xlabel("")
        ax4.set_ylabel("Delta Pearson (CPFLOW - MultiVCDiff)")
        ax4.tick_params(axis="x", rotation=90, labelsize=5)

    save_pub(fig, output_dir / "figure_model_comparison")
    plt.close(fig)

    cp_pred = pd.read_csv(pred_paths["CPFLOW_combined"], index_col=0)
    mvc_pred = pd.read_csv(pred_paths["MultiVCDiff_original"], index_col=0)
    show = deltas.sort_values(
        ["delta_pearson_cpflow_minus_multivcdiff", "pearson_CPFLOW_combined"],
        ascending=[False, False],
    ).head(showcase_n)
    show.to_csv(output_dir / "showcase_selected_drugs.csv", index=False)

    n = show.shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 2.05 * n), constrained_layout=True)
    if n == 1:
        axes = np.asarray([axes])
    for row_i, (_, row) in enumerate(show.iterrows()):
        drug = row["drug"]
        real = bundle.real_means[drug]
        cp = cp_pred.loc[drug].values.astype(float)
        mv = mvc_pred.loc[drug].values.astype(float)

        axes[row_i, 0].scatter(real, mv, s=7, alpha=0.45, color=palette["MultiVCDiff_original"], rasterized=True)
        axes[row_i, 0].scatter(real, cp, s=7, alpha=0.45, color=palette["CPFLOW_combined"], rasterized=True)
        lo = min(real.min(), cp.min(), mv.min())
        hi = max(real.max(), cp.max(), mv.max())
        axes[row_i, 0].plot([lo, hi], [lo, hi], color="#777777", lw=0.7, ls="--")
        axes[row_i, 0].set_title(
            f"{drug}: Pearson {row['pearson_MultiVCDiff_original']:.2f} -> {row['pearson_CPFLOW_combined']:.2f}",
            fontsize=8,
        )
        axes[row_i, 0].set_xlabel("Real mean RNA")
        axes[row_i, 0].set_ylabel("Predicted mean RNA")

        effect = np.abs(real - bundle.global_mean)
        top = np.argsort(effect)[-20:]
        order = top[np.argsort(real[top])]
        y = np.arange(len(order))
        axes[row_i, 1].plot(real[order], y, color="#222222", lw=1.2, label="Real")
        axes[row_i, 1].plot(mv[order], y, color=palette["MultiVCDiff_original"], lw=1.0, label="MultiVCDiff")
        axes[row_i, 1].plot(cp[order], y, color=palette["CPFLOW_combined"], lw=1.0, label="CPFLOW")
        axes[row_i, 1].set_yticks(y)
        axes[row_i, 1].set_yticklabels([bundle.gene_names[i] for i in order], fontsize=5)
        axes[row_i, 1].set_xlabel("Mean RNA")
        axes[row_i, 1].set_title("Top response genes", fontsize=8)
        if row_i == 0:
            axes[row_i, 1].legend(loc="best", fontsize=6)

    save_pub(fig, output_dir / "figure_drug_showcase")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT / "comparison_outputs")
    p.add_argument("--models", nargs="+", choices=["cpflow", "multivcdiff"], default=["cpflow", "multivcdiff"])
    p.add_argument("--h5ad-path", default="/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad")
    p.add_argument("--ctrl-rna-h5ad", default="/data1/dataset/stem_cell/CPgenes/rna_ctrl_data_filtered.h5ad")
    p.add_argument("--drug-column", default="compound")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-rna-features", type=int, default=977)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--use-ema", action="store_true", default=True)

    p.add_argument("--cpflow-ckpt", default=str(ROOT / "results_combined/000-CPFlow-B-2-noise/checkpoints/0008000.pt"))
    p.add_argument("--cpflow-model", default="CPFlow-B/2")
    p.add_argument("--cpflow-rna-tokens", type=int, default=8)
    p.add_argument("--cpflow-fusion-every", type=int, default=4)
    p.add_argument("--cpflow-source", choices=["noise", "ctrl-rna"], default="ctrl-rna")
    p.add_argument("--cpflow-gen-per-drug", type=int, default=16)
    p.add_argument("--cpflow-num-steps", type=int, default=50)
    p.add_argument("--cpflow-cfg-scale", type=float, default=1.5)
    p.add_argument("--cpflow-method", choices=["euler", "heun"], default="heun")

    p.add_argument("--mvc-ckpt", default="/data1/dataset/stem_cell/CPgenes/0100000.pt")
    p.add_argument("--mvc-model", default="DiTMultimodal-XL/2")
    p.add_argument("--mvc-num-drug-classes", type=int, default=98)
    p.add_argument("--mvc-gen-per-drug", type=int, default=16)
    p.add_argument("--mvc-num-steps", type=int, default=50)
    p.add_argument("--mvc-batch-size", type=int, default=2)

    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--showcase-n", type=int, default=4)
    p.add_argument("--max-drugs", type=int, default=None, help="Optional smoke-test limit over sorted drugs.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    bundle = load_bundle(args)
    all_rows: List[Dict] = []
    pred_paths: Dict[str, Path] = {}

    if "cpflow" in args.models:
        cp_path = args.output_dir / "pred_means_cpflow_combined.csv"
        pred_paths["CPFLOW_combined"] = cp_path
        if args.skip_existing and cp_path.exists():
            print(f"[CPFLOW] using existing {cp_path}", flush=True)
            cp_pred = pd.read_csv(cp_path, index_col=0)
            pred = {drug: cp_pred.loc[drug].values.astype(float) for drug in cp_pred.index}
        else:
            pred = eval_cpflow(args, bundle, device)
            save_pred_matrix(pred, bundle, cp_path)
        all_rows.extend(metric_rows("CPFLOW_combined", pred, bundle, args.cpflow_num_steps, args.cpflow_gen_per_drug))

    if "multivcdiff" in args.models:
        mvc_path = args.output_dir / "pred_means_multivcdiff_original.csv"
        pred_paths["MultiVCDiff_original"] = mvc_path
        if args.skip_existing and mvc_path.exists():
            print(f"[MultiVCDiff] using existing {mvc_path}", flush=True)
            mvc_pred = pd.read_csv(mvc_path, index_col=0)
            pred = {drug: mvc_pred.loc[drug].values.astype(float) for drug in mvc_pred.index}
        else:
            pred = eval_multivcdiff(args, bundle, device)
            save_pred_matrix(pred, bundle, mvc_path)
        all_rows.extend(metric_rows("MultiVCDiff_original", pred, bundle, args.mvc_num_steps, args.mvc_gen_per_drug))

    per_drug = pd.DataFrame(all_rows)
    per_drug.to_csv(args.output_dir / "per_drug_metrics.csv", index=False)
    summary = summarize(per_drug, args.output_dir)
    deltas = merge_deltas(per_drug)
    deltas.to_csv(args.output_dir / "per_drug_deltas.csv", index=False)

    if not args.no_plots and {"CPFLOW_combined", "MultiVCDiff_original"}.issubset(pred_paths):
        plot_outputs(per_drug, summary, deltas, bundle, pred_paths, args.output_dir, args.showcase_n)

    print("\n=== Model summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nOutputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
