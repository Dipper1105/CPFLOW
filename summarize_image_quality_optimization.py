from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def best_by_drug(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, group in scores.groupby("drug", sort=False):
        rows.append(group.sort_values("score", ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def metric_summary(original: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original_best = best_by_drug(original).set_index("drug")
    optimized_best = best_by_drug(optimized).set_index("drug")
    for drug in optimized_best.index:
        o = original_best.loc[drug]
        f = optimized_best.loc[drug]
        ref_fg = float(f["real_ref_foreground"])
        ref_sharp = float(f["real_ref_sharpness"])
        ref_block = float(f["real_ref_blockiness"])
        original_fg_err = abs(float(o["candidate_foreground"]) - ref_fg)
        optimized_fg_err = abs(float(f["candidate_foreground"]) - ref_fg)
        original_sharp_err = abs(np.log((float(o["candidate_sharpness"]) + 1e-8) / (ref_sharp + 1e-8)))
        optimized_sharp_err = abs(np.log((float(f["candidate_sharpness"]) + 1e-8) / (ref_sharp + 1e-8)))
        original_block_err = abs(float(o["candidate_blockiness"]) - ref_block)
        optimized_block_err = abs(float(f["candidate_blockiness"]) - ref_block)
        rows.append(
            {
                "drug": drug,
                "original_candidate": int(o["candidate"]),
                "optimized_candidate": int(f["candidate"]),
                "original_score": float(o["score"]),
                "optimized_score": float(f["score"]),
                "delta_score": float(f["score"] - o["score"]),
                "real_foreground": ref_fg,
                "original_foreground": float(o["candidate_foreground"]),
                "optimized_foreground": float(f["candidate_foreground"]),
                "delta_foreground_abs_error": float(original_fg_err - optimized_fg_err),
                "real_sharpness": ref_sharp,
                "original_sharpness": float(o["candidate_sharpness"]),
                "optimized_sharpness": float(f["candidate_sharpness"]),
                "delta_log_sharpness_abs_error": float(original_sharp_err - optimized_sharp_err),
                "real_blockiness": ref_block,
                "original_blockiness": float(o["candidate_blockiness"]),
                "optimized_blockiness": float(f["candidate_blockiness"]),
                "delta_blockiness_abs_error": float(original_block_err - optimized_block_err),
                "original_display_path": str(o["display_path"]),
                "optimized_display_path": str(f["display_path"]),
            }
        )
    return pd.DataFrame(rows)


def image_ax(ax, path: Path, title: str = "", ylabel: str | None = None) -> None:
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)
    if ylabel:
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=42, fontsize=8)


def plot_before_after_plate(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    cols = ["Real", "Original CPFLOW", "Image-finetuned CPFLOW"]
    fig, axes = plt.subplots(
        len(summary),
        len(cols),
        figsize=(5.7, 1.75 * len(summary)),
        constrained_layout=True,
    )
    if len(summary) == 1:
        axes = np.asarray([axes])

    for r, row in summary.iterrows():
        drug = row["drug"]
        real_path = args.optimized_dir / "images" / drug / "real_00.png"
        original_path = resolve_path(row["original_display_path"], args.root)
        optimized_path = resolve_path(row["optimized_display_path"], args.root)
        titles = [
            "Real" if r == 0 else "",
            f"Original #{int(row['original_candidate'])}\nscore {row['original_score']:.2f}" if r == 0 else "",
            f"Optimized #{int(row['optimized_candidate'])}\nscore {row['optimized_score']:.2f}" if r == 0 else "",
        ]
        image_ax(axes[r, 0], real_path, titles[0], ylabel=drug)
        image_ax(axes[r, 1], original_path, titles[1])
        image_ax(axes[r, 2], optimized_path, titles[2])

    stem = args.output_dir / "figure_cpflow_image_quality_before_after"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_metric_changes(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    x = np.arange(len(summary))
    width = 0.34
    colors = {"original": "#9aa4b2", "optimized": "#167a72"}
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.25), constrained_layout=True)

    axes[0].bar(x - width / 2, summary["original_score"], width, color=colors["original"], label="Original")
    axes[0].bar(x + width / 2, summary["optimized_score"], width, color=colors["optimized"], label="Optimized")
    axes[0].axhline(0, color="#222222", linewidth=0.6)
    axes[0].set_ylabel("Quality score")
    axes[0].set_title("Higher is better")

    original_fg_err = (summary["original_foreground"] - summary["real_foreground"]).abs()
    optimized_fg_err = (summary["optimized_foreground"] - summary["real_foreground"]).abs()
    axes[1].bar(x - width / 2, original_fg_err, width, color=colors["original"])
    axes[1].bar(x + width / 2, optimized_fg_err, width, color=colors["optimized"])
    axes[1].set_ylabel("|foreground - real|")
    axes[1].set_title("Lower is better")

    original_block_err = (summary["original_blockiness"] - summary["real_blockiness"]).abs()
    optimized_block_err = (summary["optimized_blockiness"] - summary["real_blockiness"]).abs()
    axes[2].bar(x - width / 2, original_block_err, width, color=colors["original"])
    axes[2].bar(x + width / 2, optimized_block_err, width, color=colors["optimized"])
    axes[2].set_ylabel("|blockiness - real|")
    axes[2].set_title("Lower is better")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(summary["drug"], rotation=30, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=6.5)

    stem = args.output_dir / "figure_cpflow_image_quality_metrics"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    mean_delta = summary["delta_score"].mean()
    lines = [
        "# CPFLOW image-quality optimization summary",
        "",
        "This summary compares morphology-aware samples from the original combined CPFLOW checkpoint",
        "against samples from the image-focused fine-tuned checkpoint.",
        "",
        f"Original candidate directory: `{args.original_dir}`",
        f"Optimized candidate directory: `{args.optimized_dir}`",
        f"Image-finetuned checkpoint: `{args.finetuned_checkpoint}`",
        "",
        f"Mean best-candidate score gain across drugs: `{mean_delta:.3f}`.",
        "",
        "| Drug | Original score | Optimized score | Delta score | Foreground error gain | Blockiness error gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['drug']} | {row['original_score']:.3f} | {row['optimized_score']:.3f} | "
            f"{row['delta_score']:.3f} | {row['delta_foreground_abs_error']:.3f} | "
            f"{row['delta_blockiness_abs_error']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            "- `figure_cpflow_image_quality_before_after.png/svg/pdf/tiff`",
            "- `figure_cpflow_image_quality_metrics.png/svg/pdf/tiff`",
            "- `image_quality_optimization_metrics.csv`",
            "",
            "Interpretation:",
            "",
            "The image-focused checkpoint produces visibly more cell-like samples than the original checkpoint,",
            "with clearer nuclei/cytoplasm separation and candidate scores closer to real-image statistics.",
            "Residual artifacts remain, especially local block texture and density mismatch, so the result should be",
            "reported as improved but not yet article-grade image synthesis.",
        ]
    )
    (args.output_dir / "image_quality_optimization_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--original-dir", type=Path, default=Path("CPFLOW/comparison_outputs/morphology_aware_cpflow"))
    p.add_argument("--optimized-dir", type=Path, default=Path("CPFLOW/comparison_outputs/morphology_aware_cpflow_imageft"))
    p.add_argument("--output-dir", type=Path, default=Path("CPFLOW/comparison_outputs/image_quality_optimization_summary"))
    p.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=Path("CPFLOW/results_image_finetune/000-CPFlow-B-2-imageft/checkpoints/last.pt"),
    )
    args = p.parse_args()
    args.original_dir = resolve_path(str(args.original_dir), args.root)
    args.optimized_dir = resolve_path(str(args.optimized_dir), args.root)
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

    original_scores = pd.read_csv(args.original_dir / "candidate_quality_scores.csv")
    optimized_scores = pd.read_csv(args.optimized_dir / "candidate_quality_scores.csv")
    summary = metric_summary(original_scores, optimized_scores)
    summary.to_csv(args.output_dir / "image_quality_optimization_metrics.csv", index=False)
    plot_before_after_plate(summary, args)
    plot_metric_changes(summary, args)
    write_report(summary, args)
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
