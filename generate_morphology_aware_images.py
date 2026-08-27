"""
Generate clearer CPFLOW cell-image showcases with morphology-aware selection.

This script does not retrain the network. It improves the usable image output
from the current checkpoint by:

1. Generating multiple image candidates for each drug.
2. Decoding at 512 px, without downsampling candidate files.
3. Applying robust channel-wise Cell Painting display normalization.
4. Ranking candidates by simple morphology-quality rules learned from real
   treated images: foreground fraction, channel statistics, sharpness, and
   grid/block artifact penalties.
5. Plotting real references and selected CPFLOW samples.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_cpflow_multivcdiff import (  # noqa: E402
    CPFlow_models,
    EvalBundle,
    FlowTransport,
    cpflow_drug_encoder,
    load_bundle,
    set_seed,
)


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    return (np.clip(arr01, 0, 1) * 255).round().astype(np.uint8)


def tensor_to_arr01(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().clamp(-1, 1)
    arr = (arr + 1.0) / 2.0
    return arr.permute(1, 2, 0).numpy()


def robust_cellpaint_display(arr01: np.ndarray) -> np.ndarray:
    """Percentile display stretch per channel, preserving dark background."""
    out = np.zeros_like(arr01, dtype=np.float32)
    for c in range(3):
        ch = arr01[..., c]
        lo, hi = np.percentile(ch, [1.0, 99.7])
        if hi <= lo + 1e-6:
            out[..., c] = np.clip(ch, 0, 1)
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    out = np.power(out, 0.85)
    return out


def image_stats(arr01: np.ndarray) -> Dict[str, float]:
    """Heuristic Cell Painting normality statistics on display-normalized RGB."""
    gray = arr01.max(axis=2)
    fg = gray > 0.14
    foreground = float(fg.mean())
    channel_mean = arr01.mean(axis=(0, 1))
    channel_std = arr01.std(axis=(0, 1))
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    sharpness = float(dx.mean() + dy.mean())
    contrast = float(np.percentile(gray, 99) - np.percentile(gray, 5))
    saturation = float((arr01.max(axis=2) - arr01.min(axis=2)).mean())
    # Penalize checkerboard/block artifacts by comparing grid-boundary changes
    # to ordinary neighboring-pixel changes.
    if gray.shape[0] >= 32 and gray.shape[1] >= 32:
        grid_y = np.arange(16, gray.shape[0], 16)
        grid_x = np.arange(16, gray.shape[1], 16)
        boundary = 0.0
        count = 0
        if len(grid_y):
            boundary += np.abs(gray[grid_y, :] - gray[grid_y - 1, :]).mean()
            count += 1
        if len(grid_x):
            boundary += np.abs(gray[:, grid_x] - gray[:, grid_x - 1]).mean()
            count += 1
        boundary = boundary / max(count, 1)
        blockiness = float(boundary / (dx.mean() + dy.mean() + 1e-6))
    else:
        blockiness = 1.0
    return {
        "foreground": foreground,
        "sharpness": sharpness,
        "contrast": contrast,
        "saturation": saturation,
        "mean_r": float(channel_mean[0]),
        "mean_g": float(channel_mean[1]),
        "mean_b": float(channel_mean[2]),
        "std_r": float(channel_std[0]),
        "std_g": float(channel_std[1]),
        "std_b": float(channel_std[2]),
        "blockiness": blockiness,
    }


def median_real_stats(args, bundle: EvalBundle, drug: str) -> Dict[str, float]:
    rows = bundle.obs[bundle.obs[args.drug_column] == drug].head(args.real_ref_count)
    stats = []
    for _, row in rows.iterrows():
        path = Path(args.image_dir) / str(row[args.image_column])
        arr = np.asarray(Image.open(path).convert("RGB").resize((args.image_size, args.image_size))).astype(np.float32) / 255.0
        stats.append(image_stats(robust_cellpaint_display(arr)))
    return {k: float(np.median([s[k] for s in stats])) for k in stats[0]}


def score_candidate(stats: Dict[str, float], ref: Dict[str, float]) -> float:
    """Higher is better. Terms favor real-like foreground/channel stats and sharpness."""
    score = 0.0
    score -= abs(stats["foreground"] - ref["foreground"]) * 3.0
    score -= abs(stats["contrast"] - ref["contrast"]) * 0.8
    score -= abs(stats["saturation"] - ref["saturation"]) * 0.8
    for c in ["r", "g", "b"]:
        score -= abs(stats[f"mean_{c}"] - ref[f"mean_{c}"]) * 1.2
        score -= abs(stats[f"std_{c}"] - ref[f"std_{c}"]) * 0.7
    # Sharp enough, but not a high-frequency noise carpet.
    sharp_ratio = stats["sharpness"] / (ref["sharpness"] + 1e-6)
    score -= abs(np.log(np.clip(sharp_ratio, 1e-3, 1e3))) * 0.5
    # Grid artifacts above real-image levels are heavily penalized.
    score -= max(0.0, stats["blockiness"] - max(1.25, ref["blockiness"] * 1.15)) * 1.0
    # Avoid blank and full-field noisy images.
    if stats["foreground"] < 0.03 or stats["foreground"] > 0.92:
        score -= 3.0
    return float(score)


def select_drugs(args) -> List[str]:
    if args.drugs:
        return args.drugs
    deltas = pd.read_csv(args.comparison_dir / "per_drug_deltas.csv")
    return deltas.sort_values("pearson_CPFLOW_combined", ascending=False)["drug"].head(args.top_n).tolist()


@torch.no_grad()
def generate_candidates(args, bundle: EvalBundle, drugs: List[str], device: torch.device):
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

    rows = []
    for drug_i, drug in enumerate(drugs, start=1):
        print(f"[drug] {drug}", flush=True)
        ref_stats = median_real_stats(args, bundle, drug)
        smi = bundle.drug_to_smiles[drug]
        fp = torch.tensor(cpflow_drug_encoder([smi], num_bits=args.fp_size), dtype=torch.float32, device=device)
        drug_dir = args.output_dir / "images" / drug
        drug_dir.mkdir(parents=True, exist_ok=True)

        # Save a small set of real references for visual comparison.
        real_rows = bundle.obs[bundle.obs[args.drug_column] == drug].head(args.real_show_count)
        for real_i, (_, row) in enumerate(real_rows.iterrows()):
            src = Path(args.image_dir) / str(row[args.image_column])
            arr = np.asarray(Image.open(src).convert("RGB").resize((args.image_size, args.image_size))).astype(np.float32) / 255.0
            Image.fromarray(to_uint8(robust_cellpaint_display(arr))).save(drug_dir / f"real_{real_i:02d}.png")

        generated = 0
        while generated < args.candidates_per_drug:
            bs = min(args.batch_size, args.candidates_per_drug - generated)
            set_seed(args.seed + 1000 * drug_i + generated)
            img0 = torch.randn(bs, 4, latent, latent, device=device)
            if bundle.ctrl_rna_norm is None:
                rna0 = torch.randn(bs, args.num_rna_features, device=device)
            else:
                choice = np.random.choice(bundle.ctrl_rna_norm.shape[0], size=bs)
                rna0 = bundle.ctrl_rna_norm[choice].to(device)
            img_lat, _ = flow.sample_ode(
                model,
                img0,
                rna0,
                fp.expand(bs, -1),
                num_steps=args.cpflow_num_steps,
                cfg_scale=args.cpflow_cfg_scale,
                method=args.cpflow_method,
            )
            decoded = vae.decode(img_lat / 0.18215).sample
            for j in range(bs):
                sample_id = generated + j
                raw01 = tensor_to_arr01(decoded[j])
                disp = robust_cellpaint_display(raw01)
                stats = image_stats(disp)
                score = score_candidate(stats, ref_stats)
                raw_path = drug_dir / f"candidate_{sample_id:03d}_raw.png"
                disp_path = drug_dir / f"candidate_{sample_id:03d}_display.png"
                Image.fromarray(to_uint8(raw01)).save(raw_path)
                Image.fromarray(to_uint8(disp)).save(disp_path)
                row = {
                    "drug": drug,
                    "candidate": sample_id,
                    "score": score,
                    "raw_path": str(raw_path),
                    "display_path": str(disp_path),
                }
                row.update({f"candidate_{k}": v for k, v in stats.items()})
                row.update({f"real_ref_{k}": v for k, v in ref_stats.items()})
                rows.append(row)
            generated += bs

    del model, vae
    torch.cuda.empty_cache()
    gc.collect()
    scores = pd.DataFrame(rows)
    scores.to_csv(args.output_dir / "candidate_quality_scores.csv", index=False)
    return scores


def plot_plate(args, drugs: List[str], scores: pd.DataFrame):
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
    cols = 1 + args.keep_per_drug
    fig, axes = plt.subplots(len(drugs), cols, figsize=(1.65 * cols, 1.65 * len(drugs)), constrained_layout=True)
    if len(drugs) == 1:
        axes = np.asarray([axes])
    for r, drug in enumerate(drugs):
        real_path = args.output_dir / "images" / drug / "real_00.png"
        best = scores[scores["drug"] == drug].sort_values("score", ascending=False).head(args.keep_per_drug)
        paths = [real_path] + [Path(p) for p in best["display_path"]]
        titles = ["Real"] + [f"CPFLOW #{int(c)}\nscore {s:.2f}" for c, s in zip(best["candidate"], best["score"])]
        for c, path in enumerate(paths):
            ax = axes[r, c]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(drug, rotation=0, ha="right", va="center", labelpad=38, fontsize=8)
    stem = args.output_dir / "figure_morphology_aware_cpflow_plate"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(args, drugs: List[str], scores: pd.DataFrame):
    lines = [
        "# Morphology-aware CPFLOW image generation",
        "",
        "This run uses the current CPFLOW checkpoint and improves image display by generating multiple candidates,",
        "ranking them against real Cell Painting statistics, and plotting the best-scoring samples.",
        "",
        f"Drugs: {', '.join(drugs)}",
        f"Candidates per drug: {args.candidates_per_drug}",
        f"Kept per drug: {args.keep_per_drug}",
        "",
        "| Drug | Best candidate | Best score | Foreground | Sharpness | Blockiness |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for drug in drugs:
        best = scores[scores["drug"] == drug].sort_values("score", ascending=False).iloc[0]
        lines.append(
            f"| {drug} | {int(best['candidate'])} | {best['score']:.3f} | "
            f"{best['candidate_foreground']:.3f} | {best['candidate_sharpness']:.3f} | "
            f"{best['candidate_blockiness']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            "- `figure_morphology_aware_cpflow_plate.png/svg/pdf`",
            "- `candidate_quality_scores.csv`",
            "- `images/<drug>/candidate_*_raw.png` and `candidate_*_display.png`",
            "",
            "Important limitation: this is morphology-aware sampling and candidate selection, not a newly retrained image model.",
            "It can select cleaner samples from the current model, but true article-level image fidelity likely requires image-focused fine-tuning.",
        ]
    )
    (args.output_dir / "morphology_aware_report.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-dir", type=Path, default=ROOT / "comparison_outputs/full_44drugs_cpflow50_mvc50")
    p.add_argument("--output-dir", type=Path, default=ROOT / "comparison_outputs/morphology_aware_cpflow")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--drugs", nargs="*", default=None)
    p.add_argument("--h5ad-path", default="/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad")
    p.add_argument("--ctrl-rna-h5ad", default="/data1/dataset/stem_cell/CPgenes/rna_ctrl_data_filtered.h5ad")
    p.add_argument("--image-dir", default="/data1/dataset/stem_cell/CPgenes/merged_rgb_images_train_all")
    p.add_argument("--image-column", default="merged_image")
    p.add_argument("--drug-column", default="compound")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-rna-features", type=int, default=977)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-drugs", type=int, default=None)
    p.add_argument("--vae-path", default="/data1/nicole/models/sd-vae-ft-ema")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--cpflow-ckpt", default=str(ROOT / "results_combined/000-CPFlow-B-2-noise/checkpoints/0008000.pt"))
    p.add_argument("--cpflow-model", default="CPFlow-B/2")
    p.add_argument("--cpflow-rna-tokens", type=int, default=8)
    p.add_argument("--cpflow-fusion-every", type=int, default=4)
    p.add_argument("--cpflow-num-steps", type=int, default=50)
    p.add_argument("--cpflow-cfg-scale", type=float, default=1.5)
    p.add_argument("--cpflow-method", choices=["euler", "heun"], default="heun")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--candidates-per-drug", type=int, default=24)
    p.add_argument("--keep-per-drug", type=int, default=3)
    p.add_argument("--real-ref-count", type=int, default=16)
    p.add_argument("--real-show-count", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    drugs = select_drugs(args)
    bundle = load_bundle(args)
    scores = generate_candidates(args, bundle, drugs, device)
    plot_plate(args, drugs, scores)
    write_report(args, drugs, scores)
    print(f"Outputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
