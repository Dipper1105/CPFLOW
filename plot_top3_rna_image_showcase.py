"""
Top-3 drug showcase for RNA metrics and predicted cell images.

Selection rule: highest CPFLOW per-drug Pearson from an existing comparison
directory. The script generates one CPFLOW and one MultiVCDiff image sample per
selected drug, adds a real treated image reference, and plots RNA Pearson plus
delta Pearson against the original model.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_cpflow_multivcdiff import (  # noqa: E402
    EvalBundle,
    FlowTransport,
    CPFlow_models,
    cpflow_drug_encoder,
    import_multivcdiff_modules,
    load_bundle,
    set_seed,
)


def pil_from_tensor(x: torch.Tensor) -> Image.Image:
    arr = x.detach().float().cpu().clamp(-1, 1)
    arr = (arr + 1.0) / 2.0
    arr = arr.permute(1, 2, 0).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_pub(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def select_top3(args) -> pd.DataFrame:
    deltas = pd.read_csv(args.comparison_dir / "per_drug_deltas.csv")
    selected = deltas.sort_values("pearson_CPFLOW_combined", ascending=False).head(args.top_n).copy()
    selected.to_csv(args.output_dir / "top3_selected_drugs.csv", index=False)
    return selected


def save_real_images(args, bundle: EvalBundle, drugs: List[str]) -> Dict[str, Path]:
    real_paths = {}
    for drug in drugs:
        rows = bundle.obs[bundle.obs[args.drug_column] == drug]
        if rows.empty:
            continue
        img_name = rows[args.image_column].iloc[0]
        src = Path(args.image_dir) / str(img_name)
        img = Image.open(src).convert("RGB").resize((args.panel_image_size, args.panel_image_size))
        dst = args.output_dir / "images" / drug / "real_reference.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        real_paths[drug] = dst
    return real_paths


@torch.no_grad()
def generate_cpflow_images(args, bundle: EvalBundle, drugs: List[str], device: torch.device) -> Dict[str, Path]:
    from diffusers.models import AutoencoderKL

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

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    flow = FlowTransport(sigma=0.0)
    latent = args.image_size // 8
    paths = {}

    for i, drug in enumerate(drugs, start=1):
        set_seed(args.seed + i)
        smi = bundle.drug_to_smiles[drug]
        fp = torch.tensor(cpflow_drug_encoder([smi], num_bits=args.fp_size), dtype=torch.float32, device=device)
        img0 = torch.randn(1, 4, latent, latent, device=device)
        if bundle.ctrl_rna_norm is None:
            rna0 = torch.randn(1, args.num_rna_features, device=device)
        else:
            choice = np.random.choice(bundle.ctrl_rna_norm.shape[0], size=1)
            rna0 = bundle.ctrl_rna_norm[choice].to(device)
        img_lat, _ = flow.sample_ode(
            model,
            img0,
            rna0,
            fp,
            num_steps=args.cpflow_num_steps,
            cfg_scale=args.cpflow_cfg_scale,
            method=args.cpflow_method,
        )
        image = vae.decode(img_lat / 0.18215).sample[0]
        dst = args.output_dir / "images" / drug / "cpflow_pred.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pil_from_tensor(image).resize((args.panel_image_size, args.panel_image_size)).save(dst)
        paths[drug] = dst
        print(f"[CPFLOW image] {drug}", flush=True)

    del model, vae
    torch.cuda.empty_cache()
    gc.collect()
    return paths


@torch.no_grad()
def generate_multivcdiff_images(args, bundle: EvalBundle, drugs: List[str], device: torch.device) -> Dict[str, Path]:
    from diffusers.models import AutoencoderKL

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

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    diffusion = diffusion_mod.create_diffusion(str(args.mvc_num_steps))
    latent = args.image_size // 8
    paths = {}

    for i, drug in enumerate(drugs, start=1):
        set_seed(args.seed + 1000 + i)
        smi = bundle.drug_to_smiles[drug]
        fp = torch.tensor(mvc_dataset.Drug_encoder([smi], num_Bits=args.fp_size), dtype=torch.float32, device=device)
        img = torch.randn(1, 4, latent, latent, device=device)
        rna = torch.randn(1, args.num_rna_features, device=device)
        model_kwargs = {"drug_fp": fp, "rna": rna}
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
            raise RuntimeError(f"No MultiVCDiff image sample for {drug}")
        image = vae.decode(final_sample["img_sample"] / 0.18215).sample[0]
        dst = args.output_dir / "images" / drug / "multivcdiff_pred.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pil_from_tensor(image).resize((args.panel_image_size, args.panel_image_size)).save(dst)
        paths[drug] = dst
        print(f"[MultiVCDiff image] {drug}", flush=True)

    del model, vae
    torch.cuda.empty_cache()
    gc.collect()
    return paths


def plot_rna(selected: pd.DataFrame, args):
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
    palette = {"CPFLOW combined": "#3B7EA1", "MultiVCDiff original": "#B45F4D"}
    rows = []
    for _, r in selected.iterrows():
        rows.append({"drug": r["drug"], "model": "CPFLOW combined", "pearson": r["pearson_CPFLOW_combined"]})
        rows.append({"drug": r["drug"], "model": "MultiVCDiff original", "pearson": r["pearson_MultiVCDiff_original"]})
    pearson_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35), constrained_layout=True)
    sns.barplot(data=pearson_df, x="drug", y="pearson", hue="model", palette=palette, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RNA Pearson")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(loc="lower right", fontsize=6)

    sns.barplot(
        data=selected,
        x="drug",
        y="delta_pearson_cpflow_minus_multivcdiff",
        color="#3B7EA1",
        ax=axes[1],
    )
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Delta Pearson\n(CPFLOW - MultiVCDiff)")
    axes[1].tick_params(axis="x", rotation=25)
    for tick, val in zip(axes[1].get_xticks(), selected["delta_pearson_cpflow_minus_multivcdiff"]):
        axes[1].text(tick, val + 0.015, f"+{val:.2f}", ha="center", va="bottom", fontsize=6)

    save_pub(fig, args.output_dir / "figure_top3_rna_pearson_delta")
    plt.close(fig)


def plot_images(selected: pd.DataFrame, image_paths: Dict[str, Dict[str, Path]], args):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
        }
    )

    drugs = selected["drug"].tolist()
    columns = [("real", "Real treated"), ("multivcdiff", "MultiVCDiff pred."), ("cpflow", "CPFLOW pred.")]
    fig, axes = plt.subplots(len(drugs), len(columns), figsize=(5.4, 1.78 * len(drugs)), constrained_layout=True)
    if len(drugs) == 1:
        axes = np.asarray([axes])

    for r, drug in enumerate(drugs):
        for c, (key, title) in enumerate(columns):
            ax = axes[r, c]
            img = Image.open(image_paths[drug][key]).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(drug, rotation=0, ha="right", va="center", labelpad=38, fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)

    save_pub(fig, args.output_dir / "figure_top3_cell_image_comparison")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-dir", type=Path, default=ROOT / "comparison_outputs/full_44drugs_cpflow50_mvc50")
    p.add_argument("--output-dir", type=Path, default=ROOT / "comparison_outputs/top3_showcase")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--h5ad-path", default="/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad")
    p.add_argument("--ctrl-rna-h5ad", default="/data1/dataset/stem_cell/CPgenes/rna_ctrl_data_filtered.h5ad")
    p.add_argument("--image-dir", default="/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all")
    p.add_argument("--image-column", default="merged_image")
    p.add_argument("--drug-column", default="compound")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--panel-image-size", type=int, default=256)
    p.add_argument("--num-rna-features", type=int, default=977)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=713)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vae-path", default="/data1/nicole/models/sd-vae-ft-ema")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--skip-existing-images", action="store_true", default=True)

    p.add_argument("--cpflow-ckpt", default=str(ROOT / "results_combined/000-CPFlow-B-2-noise/checkpoints/0008000.pt"))
    p.add_argument("--cpflow-model", default="CPFlow-B/2")
    p.add_argument("--cpflow-rna-tokens", type=int, default=8)
    p.add_argument("--cpflow-fusion-every", type=int, default=4)
    p.add_argument("--cpflow-num-steps", type=int, default=50)
    p.add_argument("--cpflow-cfg-scale", type=float, default=1.5)
    p.add_argument("--cpflow-method", choices=["euler", "heun"], default="heun")

    p.add_argument("--mvc-ckpt", default="/data1/dataset/stem_cell/CPgenes/0100000.pt")
    p.add_argument("--mvc-model", default="DiTMultimodal-XL/2")
    p.add_argument("--mvc-num-drug-classes", type=int, default=98)
    p.add_argument("--mvc-num-steps", type=int, default=50)
    p.add_argument("--max-drugs", type=int, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    selected = select_top3(args)
    drugs = selected["drug"].tolist()
    bundle = load_bundle(args)

    image_paths: Dict[str, Dict[str, Path]] = {drug: {} for drug in drugs}
    real = save_real_images(args, bundle, drugs)
    for drug, path in real.items():
        image_paths[drug]["real"] = path

    need_cpflow = [
        drug for drug in drugs
        if not (args.skip_existing_images and (args.output_dir / "images" / drug / "cpflow_pred.png").exists())
    ]
    if need_cpflow:
        cp_paths = generate_cpflow_images(args, bundle, need_cpflow, device)
    else:
        cp_paths = {}
    for drug in drugs:
        image_paths[drug]["cpflow"] = cp_paths.get(drug, args.output_dir / "images" / drug / "cpflow_pred.png")

    need_mvc = [
        drug for drug in drugs
        if not (args.skip_existing_images and (args.output_dir / "images" / drug / "multivcdiff_pred.png").exists())
    ]
    if need_mvc:
        mvc_paths = generate_multivcdiff_images(args, bundle, need_mvc, device)
    else:
        mvc_paths = {}
    for drug in drugs:
        image_paths[drug]["multivcdiff"] = mvc_paths.get(drug, args.output_dir / "images" / drug / "multivcdiff_pred.png")

    plot_rna(selected, args)
    plot_images(selected, image_paths, args)

    print("Selected top drugs:", ", ".join(drugs), flush=True)
    print(f"Outputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
