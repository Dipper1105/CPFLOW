from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parent


DRUG_BIOLOGY = {
    "nystatin": "Membrane sterol binding; membrane stress/permeability phenotype.",
    "puromycin": "Translation inhibition; proteotoxic stress and reduced growth/protein synthesis phenotype.",
    "pd-98059": "MEK/ERK pathway inhibition; altered proliferation and cytoskeletal signalling.",
    "hydroxyurea": "Ribonucleotide reductase inhibition; S-phase arrest and DNA-replication stress.",
    "quercetin": "Polyphenol kinase/oxidative-stress modulator; mitochondrial/redox and cytoskeletal effects.",
    "cyclophosphamide": "Alkylating DNA-damage agent; genotoxic stress and growth arrest.",
}


METRIC_LABELS = {
    "nuclear_area_fraction": "Nuclear area fraction",
    "cell_area_fraction": "Cell area fraction",
    "nucleus_to_cytoplasm_area": "N:C area ratio",
    "cytoplasm_to_nucleus_intensity": "Cytoplasm:nucleus intensity",
    "mitochondrial_brightness_proxy": "Mitochondrial brightness proxy",
    "mitochondrial_enrichment_proxy": "Mitochondrial enrichment proxy",
    "green_cytoplasm_brightness": "Cytoplasm green brightness",
    "red_cytoplasm_brightness": "Cytoplasm red brightness",
    "nuclear_brightness": "Nuclear brightness",
    "cell_count_proxy": "Cell count proxy",
    "mean_cell_area_proxy": "Mean cell area proxy",
    "clustering_proxy": "Cell clustering proxy",
    "edge_texture": "Edge/texture strength",
    "local_contrast": "Local contrast",
}


METRIC_GROUPS = {
    "Nuclear morphology": ["nuclear_area_fraction", "nucleus_to_cytoplasm_area", "nuclear_brightness"],
    "Mito/cytoplasm signal": [
        "mitochondrial_brightness_proxy",
        "mitochondrial_enrichment_proxy",
        "green_cytoplasm_brightness",
        "red_cytoplasm_brightness",
        "cytoplasm_to_nucleus_intensity",
    ],
    "Cell density/organization": ["cell_area_fraction", "cell_count_proxy", "mean_cell_area_proxy", "clustering_proxy"],
    "Texture": ["edge_texture", "local_contrast"],
}


DRUG_FOCUSED_METRICS = {
    "nystatin": {
        "process": "membrane sterol binding / membrane stress",
        "metrics": ["mitochondrial_brightness_proxy", "cell_area_fraction", "clustering_proxy"],
    },
    "puromycin": {
        "process": "translation inhibition / growth and proteotoxic stress",
        "metrics": ["cell_count_proxy", "mean_cell_area_proxy", "edge_texture"],
    },
    "pd-98059": {
        "process": "MEK-ERK inhibition / proliferation-cytoskeleton signalling",
        "metrics": ["cell_count_proxy", "cell_area_fraction", "edge_texture"],
    },
    "hydroxyurea": {
        "process": "S-phase arrest / DNA replication stress",
        "metrics": ["nucleus_to_cytoplasm_area", "nuclear_area_fraction", "cell_count_proxy"],
    },
    "quercetin": {
        "process": "redox-mitochondrial and cytoskeletal stress",
        "metrics": ["mitochondrial_brightness_proxy", "mitochondrial_enrichment_proxy", "edge_texture"],
    },
    "cyclophosphamide": {
        "process": "alkylating DNA damage / growth arrest",
        "metrics": ["nucleus_to_cytoplasm_area", "cell_count_proxy", "local_contrast"],
    },
}


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def robust_threshold(channel: np.ndarray, floor: float = 0.08) -> np.ndarray:
    smooth = ndi.gaussian_filter(channel, sigma=1.0)
    positive = smooth[smooth > floor]
    if positive.size < 128:
        thr = np.percentile(smooth, 75)
    else:
        thr = np.percentile(positive, 55)
    return smooth > max(floor, thr)


def clean_mask(mask: np.ndarray, min_size: int = 24, close_iter: int = 1) -> np.ndarray:
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=close_iter)
    labels, nlab = ndi.label(mask)
    if nlab == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labels]


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def connected_component_stats(mask: np.ndarray) -> tuple[int, float, float]:
    labels, nlab = ndi.label(mask)
    if nlab == 0:
        return 0, 0.0, 0.0
    areas = np.bincount(labels.ravel())[1:]
    return int(nlab), float(np.mean(areas)), float(np.median(areas))


def phenotype_metrics(arr: np.ndarray) -> dict[str, float]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.max(axis=2)
    cell_mask = clean_mask(gray > max(0.10, np.percentile(gray, 62)), min_size=64, close_iter=2)
    nucleus_mask = clean_mask(robust_threshold(b, floor=0.10), min_size=12, close_iter=1)
    nucleus_mask &= cell_mask | ndi.binary_dilation(cell_mask, iterations=2)
    cyto_mask = cell_mask & ~ndi.binary_dilation(nucleus_mask, iterations=1)
    bg_mask = ~cell_mask

    count, mean_area, median_area = connected_component_stats(nucleus_mask)
    cell_area = float(cell_mask.mean())
    nuclear_area = float(nucleus_mask.mean())
    cytoplasm_area = float(cyto_mask.mean())
    total_area = arr.shape[0] * arr.shape[1]
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    fg_intensity = gray[cell_mask]
    bg_intensity = gray[bg_mask]

    nuclear_b = safe_mean(b[nucleus_mask])
    cyto_mean = safe_mean(gray[cyto_mask])
    red_cyto = safe_mean(r[cyto_mask])
    green_cyto = safe_mean(g[cyto_mask])
    red_bg = safe_mean(r[bg_mask])
    mito_bright = red_cyto
    mito_enrich = red_cyto / (safe_mean(r[nucleus_mask]) + 1e-6)

    # Spatial organization: larger connected cell fields mean stronger clustering/colony-like growth.
    _, mean_cell_area, _ = connected_component_stats(cell_mask)
    clustering = mean_cell_area / (total_area + 1e-6)
    local_contrast = float(np.percentile(fg_intensity, 95) - np.percentile(fg_intensity, 10)) if fg_intensity.size else 0.0
    background_leak = safe_mean(bg_intensity)

    return {
        "nuclear_area_fraction": nuclear_area,
        "cell_area_fraction": cell_area,
        "nucleus_to_cytoplasm_area": nuclear_area / (cytoplasm_area + 1e-6),
        "cytoplasm_to_nucleus_intensity": cyto_mean / (nuclear_b + 1e-6),
        "mitochondrial_brightness_proxy": mito_bright,
        "mitochondrial_enrichment_proxy": mito_enrich,
        "green_cytoplasm_brightness": green_cyto,
        "red_cytoplasm_brightness": red_cyto,
        "nuclear_brightness": nuclear_b,
        "cell_count_proxy": count / (total_area / (512 * 512)),
        "mean_cell_area_proxy": mean_area / (total_area + 1e-6),
        "clustering_proxy": clustering,
        "edge_texture": float(dx.mean() + dy.mean()),
        "local_contrast": local_contrast,
        "background_leak": background_leak,
        "raw_nucleus_count": float(count),
        "raw_mean_nucleus_area": mean_area,
        "raw_median_nucleus_area": median_area,
        "raw_mean_cell_component_area": mean_cell_area,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    a = a[valid]
    b = b[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def metric_similarity(real: np.ndarray, gen: np.ndarray) -> float:
    valid = np.isfinite(real) & np.isfinite(gen)
    if valid.sum() == 0:
        return float("nan")
    rel_err = np.abs(gen[valid] - real[valid]) / (np.abs(real[valid]) + 1e-6)
    return float(np.exp(-np.median(rel_err)))


def save_figure(fig: plt.Figure, stem: Path, args: argparse.Namespace) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    if not args.skip_tiff:
        fig.savefig(stem.with_suffix(".tiff"), dpi=args.tiff_dpi, bbox_inches="tight")


def build_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    top = pd.read_csv(args.top_csv).head(args.top_n)
    rows = []
    for _, row in top.iterrows():
        drug = row["drug"]
        real_path = args.run_dir / "images" / drug / "real_00.png"
        gen_path = resolve_path(row["display_path"], args.root)
        for source, path in [("Real", real_path), ("Generated", gen_path)]:
            metrics = phenotype_metrics(read_rgb(path))
            out = {"drug": drug, "source": source, "path": str(path)}
            out.update(metrics)
            rows.append(out)
    metrics_df = pd.DataFrame(rows)

    sim_rows = []
    metric_cols = [m for group in METRIC_GROUPS.values() for m in group]
    for drug in top["drug"]:
        real = metrics_df[(metrics_df.drug == drug) & (metrics_df.source == "Real")].iloc[0]
        gen = metrics_df[(metrics_df.drug == drug) & (metrics_df.source == "Generated")].iloc[0]
        real_vec = real[metric_cols].astype(float).to_numpy()
        gen_vec = gen[metric_cols].astype(float).to_numpy()
        sim_row = {
            "drug": drug,
            "phenotype_cosine_similarity": cosine_similarity(real_vec, gen_vec),
            "median_relative_similarity": metric_similarity(real_vec, gen_vec),
            "biology_note": DRUG_BIOLOGY.get(drug, ""),
        }
        for metric in metric_cols:
            sim_row[f"{metric}_real"] = float(real[metric])
            sim_row[f"{metric}_generated"] = float(gen[metric])
            sim_row[f"{metric}_relative_error"] = abs(float(gen[metric]) - float(real[metric])) / (abs(float(real[metric])) + 1e-6)
        sim_rows.append(sim_row)
    sim_df = pd.DataFrame(sim_rows)
    return metrics_df, sim_df


def build_focused_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug in metrics_df["drug"].drop_duplicates():
        spec = DRUG_FOCUSED_METRICS.get(drug)
        if spec is None:
            continue
        real = metrics_df[(metrics_df.drug == drug) & (metrics_df.source == "Real")].iloc[0]
        gen = metrics_df[(metrics_df.drug == drug) & (metrics_df.source == "Generated")].iloc[0]
        for metric in spec["metrics"]:
            real_value = float(real[metric])
            gen_value = float(gen[metric])
            delta = gen_value - real_value
            pct_delta = 100.0 * delta / (abs(real_value) + 1e-6)
            rel_err = abs(delta) / (abs(real_value) + 1e-6)
            rows.append(
                {
                    "drug": drug,
                    "biological_process": spec["process"],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "real_value": real_value,
                    "generated_value": gen_value,
                    "generated_real_ratio": gen_value / (real_value + 1e-6),
                    "delta_generated_minus_real": delta,
                    "percent_delta_generated_vs_real": pct_delta,
                    "agreement_score": math.exp(-min(rel_err, 3.0)),
                }
            )
    return pd.DataFrame(rows)


def plot_similarity(sim_df: pd.DataFrame, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.1), constrained_layout=True)
    x = np.arange(sim_df.shape[0])
    ax.bar(x, sim_df["median_relative_similarity"], color="#167a72", width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(sim_df["drug"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Phenotype similarity")
    ax.set_title("Generated vs real image-derived phenotype")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stem = args.output_dir / "figure_top6_phenotype_similarity"
    save_figure(fig, stem, args)
    plt.close(fig)


def plot_metric_heatmap(sim_df: pd.DataFrame, args: argparse.Namespace) -> None:
    metric_cols = [m for group in METRIC_GROUPS.values() for m in group]
    mat = []
    for metric in metric_cols:
        vals = []
        for _, row in sim_df.iterrows():
            err = row[f"{metric}_relative_error"]
            vals.append(math.exp(-min(err, 3.0)))
        mat.append(vals)
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(sim_df.shape[0]))
    ax.set_xticklabels(sim_df["drug"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(metric_cols)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in metric_cols], fontsize=6)
    ax.set_title("Per-metric generated/real agreement")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Agreement score")
    stem = args.output_dir / "figure_top6_phenotype_metric_heatmap"
    save_figure(fig, stem, args)
    plt.close(fig)


def plot_biology_metrics(metrics_df: pd.DataFrame, args: argparse.Namespace) -> None:
    selected_metrics = [
        "mitochondrial_brightness_proxy",
        "nucleus_to_cytoplasm_area",
        "cell_count_proxy",
        "clustering_proxy",
        "edge_texture",
        "cytoplasm_to_nucleus_intensity",
    ]
    drugs = metrics_df["drug"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), constrained_layout=True)
    colors = {"Real": "#8d99a6", "Generated": "#167a72"}
    x = np.arange(len(drugs))
    width = 0.36
    for ax, metric in zip(axes.ravel(), selected_metrics):
        real_vals = []
        gen_vals = []
        for drug in drugs:
            real_vals.append(float(metrics_df[(metrics_df.drug == drug) & (metrics_df.source == "Real")][metric].iloc[0]))
            gen_vals.append(float(metrics_df[(metrics_df.drug == drug) & (metrics_df.source == "Generated")][metric].iloc[0]))
        ax.bar(x - width / 2, real_vals, width, color=colors["Real"], label="Real")
        ax.bar(x + width / 2, gen_vals, width, color=colors["Generated"], label="Generated")
        ax.set_title(METRIC_LABELS[metric], fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(drugs, rotation=35, ha="right", fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[0].legend(fontsize=6, loc="best")
    stem = args.output_dir / "figure_top6_biology_metrics_real_vs_generated"
    save_figure(fig, stem, args)
    plt.close(fig)


def plot_focused_metrics(focused_df: pd.DataFrame, args: argparse.Namespace) -> None:
    drugs = focused_df["drug"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    for ax, drug in zip(axes.ravel(), drugs):
        sub = focused_df[focused_df["drug"] == drug].copy()
        sub["plot_ratio"] = sub["generated_real_ratio"].clip(0, 2.0)
        y = np.arange(sub.shape[0])
        ax.axvline(1.0, color="#6f7d8c", lw=0.9, ls="--", zorder=0)
        for yi, ratio, pct, agreement in zip(
            y,
            sub["plot_ratio"],
            sub["percent_delta_generated_vs_real"],
            sub["agreement_score"],
        ):
            color = "#167a72" if agreement >= 0.8 else "#c76f2d"
            ax.plot([1.0, ratio], [yi, yi], color=color, lw=1.6, solid_capstyle="round")
            ax.scatter([ratio], [yi], s=28, color=color, zorder=3)
            ax.text(
                min(1.96, max(0.04, ratio + (0.05 if ratio < 1 else -0.05))),
                yi + 0.23,
                f"{pct:+.0f}%",
                ha="left" if ratio < 1 else "right",
                va="center",
                fontsize=5.5,
                color=color,
            )
        ax.scatter(np.ones_like(y), y, s=18, color="#6f7d8c", zorder=2, label="Real")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["metric_label"], fontsize=5.8)
        ax.set_xlim(0, 2.05)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xlabel("Generated / real")
        ax.set_title(drug, fontsize=8)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[len(drugs) :]:
        ax.axis("off")
    fig.suptitle("Drug-process focused phenotype preservation", fontsize=9)
    stem = args.output_dir / "figure_top6_drug_process_focused_metrics"
    save_figure(fig, stem, args)
    plt.close(fig)


def write_report(
    metrics_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    focused_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Top6 image-derived biological phenotype metrics",
        "",
        "This analysis computes lightweight biological phenotype proxies from the displayed RGB Cell Painting images.",
        "The same metrics are computed for the real perturbation image and the best generated image for each drug.",
        "",
        "Important interpretation note: the available files are RGB composites rather than separated Cell Painting",
        "channels. Therefore mitochondrial brightness is reported as a red-channel/cytoplasmic signal proxy,",
        "and nuclear/cytoplasmic regions are estimated by intensity-based masks rather than single-cell segmentation.",
        "",
        "| Drug | Phenotype similarity | Biology/process context |",
        "|---|---:|---|",
    ]
    for _, row in sim_df.iterrows():
        lines.append(f"| {row['drug']} | {row['median_relative_similarity']:.3f} | {row['biology_note']} |")
    lines.extend(
        [
            "",
            "Drug-process focused metrics:",
            "",
            "| Drug | Process | Focused metrics | Mean focused agreement |",
            "|---|---|---|---:|",
        ]
    )
    for drug in focused_df["drug"].drop_duplicates():
        sub = focused_df[focused_df["drug"] == drug]
        labels = "; ".join(sub["metric_label"].tolist())
        lines.append(
            f"| {drug} | {sub['biological_process'].iloc[0]} | {labels} | {sub['agreement_score'].mean():.3f} |"
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            "- `top6_biological_phenotype_metrics_long.csv`",
            "- `top6_biological_phenotype_similarity.csv`",
            "- `top6_drug_process_focused_metrics.csv`",
            "- `figure_top6_phenotype_similarity.png/svg/pdf`",
            "- `figure_top6_phenotype_metric_heatmap.png/svg/pdf`",
            "- `figure_top6_biology_metrics_real_vs_generated.png/svg/pdf`",
            "- `figure_top6_drug_process_focused_metrics.png/svg/pdf`",
            "",
            "TIFF export can be enabled by rerunning the script with `--write-tiff`.",
            "",
            "Metric definitions:",
            "",
            "- Mitochondrial brightness proxy: mean red-channel intensity in the estimated cytoplasmic mask.",
            "- N:C area ratio: estimated nuclear mask area divided by cytoplasmic mask area.",
            "- Cytoplasm:nucleus intensity: estimated cytoplasmic RGB intensity divided by nuclear blue-channel intensity.",
            "- Cell count proxy: connected nuclear-object count normalized to a 512 x 512 field.",
            "- Clustering proxy: mean connected cell-field area divided by image area.",
            "- Edge/texture strength: mean local absolute gradient of the RGB max-intensity image.",
        ]
    )
    (args.output_dir / "top6_biological_phenotype_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT.parent)
    p.add_argument(
        "--top-csv",
        type=Path,
        default=ROOT / "comparison_outputs/best6_generated_perturbation_images_imageft2/best6_generated_perturbation_images.csv",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "comparison_outputs/full44_imageft2_morphology_candidates",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "comparison_outputs/top6_biological_phenotypes_imageft2",
    )
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--skip-tiff", action="store_true", default=True)
    p.add_argument("--write-tiff", dest="skip_tiff", action="store_false")
    p.add_argument("--tiff-dpi", type=int, default=300)
    args = p.parse_args()
    args.top_csv = resolve_path(str(args.top_csv), args.root)
    args.run_dir = resolve_path(str(args.run_dir), args.root)
    args.output_dir = resolve_path(str(args.output_dir), args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    metrics_df, sim_df = build_metrics(args)
    focused_df = build_focused_metrics(metrics_df)
    metrics_df.to_csv(args.output_dir / "top6_biological_phenotype_metrics_long.csv", index=False)
    sim_df.to_csv(args.output_dir / "top6_biological_phenotype_similarity.csv", index=False)
    focused_df.to_csv(args.output_dir / "top6_drug_process_focused_metrics.csv", index=False)
    plot_similarity(sim_df, args)
    plot_metric_heatmap(sim_df, args)
    plot_biology_metrics(metrics_df, args)
    plot_focused_metrics(focused_df, args)
    write_report(metrics_df, sim_df, focused_df, args)
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
