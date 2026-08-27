from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent

METRIC_LABELS = {
    "nuclear_area_fraction": "Nuclear area",
    "cell_area_fraction": "Cell area",
    "nucleus_to_cytoplasm_area": "N:C area",
    "cytoplasm_to_nucleus_intensity": "Cyto:nucleus",
    "mitochondrial_brightness_proxy": "Mito brightness",
    "mitochondrial_enrichment_proxy": "Mito enrichment",
    "green_cytoplasm_brightness": "Green cyto",
    "red_cytoplasm_brightness": "Red cyto",
    "nuclear_brightness": "Nuclear brightness",
    "cell_count_proxy": "Cell count",
    "mean_cell_area_proxy": "Mean nuclear area",
    "clustering_proxy": "Cell clustering",
    "edge_texture": "Texture",
    "local_contrast": "Local contrast",
}

HEATMAP_METRICS = [
    "nuclear_area_fraction",
    "nucleus_to_cytoplasm_area",
    "nuclear_brightness",
    "mitochondrial_brightness_proxy",
    "mitochondrial_enrichment_proxy",
    "green_cytoplasm_brightness",
    "red_cytoplasm_brightness",
    "cytoplasm_to_nucleus_intensity",
    "cell_area_fraction",
    "cell_count_proxy",
    "mean_cell_area_proxy",
    "clustering_proxy",
    "edge_texture",
    "local_contrast",
]

def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        clip_on=False,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=2.5)


def draw_image_plate(fig: plt.Figure, outer: mpl.gridspec.SubplotSpec, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    grid = outer.subgridspec(2, selected.shape[0], wspace=0.04, hspace=0.04)
    first_ax = None
    for col_i, (_, row) in enumerate(selected.iterrows()):
        drug = row["drug"]
        paths = [
            args.run_dir / "images" / drug / "real_00.png",
            resolve_path(row["display_path"], args.root),
        ]
        for row_i, (source, path) in enumerate(zip(["Real", "Predict"], paths)):
            ax = fig.add_subplot(grid[row_i, col_i])
            if first_ax is None:
                first_ax = ax
            ax.imshow(read_image(path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_i == 0:
                ax.set_title(drug, fontsize=9, pad=2.0)
            if col_i == 0:
                ax.set_ylabel(source, rotation=0, ha="right", va="center", labelpad=27, fontsize=8.5)
    if first_ax is not None:
        panel_label(first_ax, "a", x=-0.34, y=1.18)
        first_ax.text(
            -0.22,
            1.18,
            "Real and Predict perturbation images",
            transform=first_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )


def draw_similarity(ax: plt.Axes, sim_df: pd.DataFrame) -> None:
    drugs = sim_df["drug"].tolist()
    vals = sim_df["median_relative_similarity"].to_numpy()
    x = np.arange(len(drugs))
    ax.bar(x, vals, color="#167a72", width=0.66)
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Phenotype similarity")
    ax.set_title("Predict vs real phenotype similarity", fontsize=9)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    style_axis(ax)


def draw_heatmap(ax: plt.Axes, sim_df: pd.DataFrame) -> None:
    mat = []
    for metric in HEATMAP_METRICS:
        row = []
        for _, drug_row in sim_df.iterrows():
            err = float(drug_row[f"{metric}_relative_error"])
            row.append(math.exp(-min(err, 3.0)))
        mat.append(row)
    mat = np.asarray(mat)
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(sim_df.shape[0]))
    ax.set_xticklabels(sim_df["drug"], rotation=35, ha="right", fontsize=6.2)
    ax.set_yticks(np.arange(len(HEATMAP_METRICS)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in HEATMAP_METRICS], fontsize=6.2)
    ax.set_title("Per-metric predict/real agreement", fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Agreement", fontsize=7)
    cb.ax.tick_params(labelsize=6.2, length=2)


def draw_focused_detail_grid(fig: plt.Figure, outer: mpl.gridspec.SubplotSpec, focused_df: pd.DataFrame) -> np.ndarray:
    sub = outer.subgridspec(2, 3, wspace=0.58, hspace=0.78)
    axes = np.asarray([fig.add_subplot(sub[i, j]) for i in range(2) for j in range(3)])
    colors = ["#167a72", "#7293a8", "#c76f2d"]
    for ax, drug in zip(axes, focused_df["drug"].drop_duplicates()):
        drug_df = focused_df[focused_df["drug"] == drug].reset_index(drop=True)
        drug_df["short_label"] = drug_df["metric"].map(METRIC_LABELS).fillna(drug_df["metric_label"])
        y = np.arange(drug_df.shape[0])
        ax.axvline(1.0, color="#6f7d8c", lw=0.8, ls="--", zorder=0)
        for idx, row in drug_df.iterrows():
            ratio = float(row["generated_real_ratio"])
            clipped = min(max(ratio, 0.0), 2.0)
            ax.plot([1.0, clipped], [idx, idx], color=colors[idx], lw=1.4, solid_capstyle="round")
            ax.scatter(clipped, idx, s=22, color=colors[idx], zorder=3)
            offset = 0.06 if clipped <= 1 else -0.06
            ax.text(
                clipped + offset,
                idx + 0.22,
                f"{float(row['percent_delta_generated_vs_real']):+.0f}%",
                ha="left" if clipped <= 1 else "right",
                va="center",
                fontsize=6.0,
                color=colors[idx],
            )
        ax.set_yticks(y)
        ax.set_yticklabels(drug_df["short_label"], fontsize=6.8)
        ax.set_xlim(0, 2.05)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.tick_params(axis="x", labelsize=6.2)
        ax.set_xlabel("Predict / real", fontsize=7.0)
        ax.set_title(drug, fontsize=8.2, pad=1.5)
        ax.invert_yaxis()
        style_axis(ax)
    axes[1].text(
        0.5,
        1.32,
        "Drug-process focused phenotype preservation",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=9.4,
        clip_on=False,
    )
    return axes


def draw_focused(ax: plt.Axes, focused_df: pd.DataFrame) -> None:
    summary = (
        focused_df.groupby("drug", sort=False)["agreement_score"]
        .mean()
        .reset_index(name="focused_agreement")
    )
    y = np.arange(summary.shape[0])
    ax.barh(y, summary["focused_agreement"], color="#167a72", height=0.64)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["drug"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean focused agreement")
    ax.set_title("Drug-process focused preservation", fontsize=9)
    for yi, val in zip(y, summary["focused_agreement"]):
        ax.text(min(val + 0.025, 0.98), yi, f"{val:.2f}", ha="left", va="center", fontsize=7)
    style_axis(ax)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT.parent)
    p.add_argument(
        "--best6-csv",
        type=Path,
        default=ROOT / "comparison_outputs/best6_generated_perturbation_images_imageft2/best6_generated_perturbation_images.csv",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "comparison_outputs/full44_imageft2_morphology_candidates",
    )
    p.add_argument(
        "--phenotype-dir",
        type=Path,
        default=ROOT / "comparison_outputs/top6_biological_phenotypes_imageft2",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "comparison_outputs/top6_combined_horizontal_figure",
    )
    args = p.parse_args()

    args.best6_csv = resolve_path(str(args.best6_csv), args.root)
    args.run_dir = resolve_path(str(args.run_dir), args.root)
    args.phenotype_dir = resolve_path(str(args.phenotype_dir), args.root)
    args.output_dir = resolve_path(str(args.output_dir), args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    selected = pd.read_csv(args.best6_csv)
    sim_df = pd.read_csv(args.phenotype_dir / "top6_biological_phenotype_similarity.csv")
    focused_df = pd.read_csv(args.phenotype_dir / "top6_drug_process_focused_metrics.csv")

    fig = plt.figure(figsize=(17.6, 8.2), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 1.15],
        hspace=0.18,
        left=0.035,
        right=0.992,
        top=0.965,
        bottom=0.085,
    )
    draw_image_plate(fig, outer[0], selected, args)

    bottom = outer[1].subgridspec(1, 4, width_ratios=[1.05, 1.2, 2.75, 1.16], wspace=0.46)
    ax_b = fig.add_subplot(bottom[0])
    ax_c = fig.add_subplot(bottom[1])
    ax_e = fig.add_subplot(bottom[3])

    draw_similarity(ax_b, sim_df)
    draw_heatmap(ax_c, sim_df)
    d_axes = draw_focused_detail_grid(fig, bottom[2], focused_df)
    draw_focused(ax_e, focused_df)
    panel_label(ax_b, "b", x=-0.18, y=1.08)
    panel_label(ax_c, "c", x=-0.15, y=1.08)
    panel_label(d_axes[0], "d", x=-0.34, y=1.32)
    panel_label(ax_e, "e", x=-0.16, y=1.08)

    stem = args.output_dir / "figure_top6_combined_horizontal_real_predict_phenotypes"
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Top6 combined horizontal figure",
        "",
        "The final horizontal figure uses a two-row layout: the first row contains Real/Predict images",
        "for the six selected drugs. Panel d is redrawn from `top6_drug_process_focused_metrics.csv`",
        "so all text uses the same font family and sizing as the rest of the figure.",
        "",
        "Candidate-specific generated labels were removed, including quercetin candidate #14 and",
        "cyclophosphamide candidate #9.",
        "",
        "Generated files:",
        "",
        "- `figure_top6_combined_horizontal_real_predict_phenotypes.png`",
        "- `figure_top6_combined_horizontal_real_predict_phenotypes.pdf`",
        "- `figure_top6_combined_horizontal_real_predict_phenotypes.svg`",
    ]
    (args.output_dir / "combined_horizontal_figure_report.md").write_text("\n".join(report) + "\n")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
