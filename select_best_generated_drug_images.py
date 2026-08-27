from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def best_candidate_per_drug(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, group in scores.groupby("drug", sort=False):
        rows.append(group.sort_values("score", ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def add_selection_metrics(best: pd.DataFrame) -> pd.DataFrame:
    best = best.copy()
    best["foreground_abs_error"] = (best["candidate_foreground"] - best["real_ref_foreground"]).abs()
    best["sharpness_log_abs_error"] = np.abs(
        np.log((best["candidate_sharpness"] + 1e-8) / (best["real_ref_sharpness"] + 1e-8))
    )
    best["blockiness_abs_error"] = (best["candidate_blockiness"] - best["real_ref_blockiness"]).abs()
    best["contrast_abs_error"] = (best["candidate_contrast"] - best["real_ref_contrast"]).abs()
    best["saturation_abs_error"] = (best["candidate_saturation"] - best["real_ref_saturation"]).abs()
    return best


def plot_selected_plate(selected: pd.DataFrame, args: argparse.Namespace) -> None:
    cols = 1 + args.keep_per_drug
    if args.keep_per_drug == 1:
        fig, axes = plt.subplots(
            len(selected),
            cols,
            figsize=(3.25, 1.28 * len(selected)),
            gridspec_kw={"wspace": 0.06, "hspace": 0.24},
        )
        fig.subplots_adjust(left=0.24, right=0.99, top=0.965, bottom=0.02)
    else:
        fig, axes = plt.subplots(
            len(selected),
            cols,
            figsize=(1.8 * cols, 1.85 * len(selected)),
            constrained_layout=True,
        )
    if len(selected) == 1:
        axes = np.asarray([axes])

    for r, (_, row) in enumerate(selected.iterrows()):
        drug = row["drug"]
        drug_scores = args.scores[args.scores["drug"] == drug].sort_values("score", ascending=False).head(args.keep_per_drug)
        paths = [args.run_dir / "images" / drug / "real_00.png"]
        paths.extend(resolve_path(p, args.root) for p in drug_scores["display_path"])
        titles = ["Real"]
        titles.extend(f"Generated #{int(c)}\nscore {s:.2f}" for c, s in zip(drug_scores["candidate"], drug_scores["score"]))

        for c, path in enumerate(paths):
            ax = axes[r, c]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0 and r == 0:
                ax.set_title(titles[c], fontsize=7.5, pad=3)
            elif c > 0:
                ax.set_title(titles[c], fontsize=6.5, pad=3)
            if c == 0:
                ax.set_ylabel(drug, rotation=0, ha="right", va="center", labelpad=42, fontsize=8)

    stem = args.output_dir / f"figure_best{args.top_n}_generated_perturbation_images"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_all_drug_ranking(best: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    plot_df = best.sort_values("score", ascending=True).reset_index(drop=True)
    colors = np.where(plot_df["drug"].isin(selected["drug"]), "#167a72", "#9aa4b2")
    fig, ax = plt.subplots(figsize=(5.0, 6.4), constrained_layout=True)
    ax.barh(plot_df["drug"], plot_df["score"], color=colors, height=0.72)
    ax.axvline(0, color="#222222", linewidth=0.6)
    ax.set_xlabel("Best generated-image quality score")
    ax.set_ylabel("")
    ax.set_title("Best candidate per drug across all 44 perturbations", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=5.8)
    stem = args.output_dir / "figure_all44_generated_image_quality_ranking"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(best: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    candidates_per_drug = args.scores.groupby("drug").size()
    score_sources = sorted(args.scores["source_csv"].unique().tolist()) if "source_csv" in args.scores else []
    lines = [
        f"# Best {args.top_n} generated perturbation images across 44 drugs",
        "",
        "The optimized image-finetuned CPFLOW checkpoint was sampled across all drugs, with an additional",
        "high-candidate search for the strongest initial image-quality hits when multiple score files are merged.",
        f"Candidates were ranked against real Cell Painting image statistics. The {args.top_n} drugs below are the",
        "globally best best-candidate examples among all generated perturbation images.",
        "",
        f"Input run directory: `{args.run_dir}`",
        f"Total drugs evaluated: `{best.shape[0]}`",
        f"Total candidates scored: `{args.scores.shape[0]}`",
        f"Candidates per drug detected: `{candidates_per_drug.median():.0f}` median, `{candidates_per_drug.max():.0f}` max",
        "",
        "| Rank | Drug | Candidate | Score | Foreground | Real foreground | Sharpness | Real sharpness | Blockiness | Real blockiness |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['drug']} | {int(row['candidate'])} | {row['score']:.3f} | "
            f"{row['candidate_foreground']:.3f} | {row['real_ref_foreground']:.3f} | "
            f"{row['candidate_sharpness']:.3f} | {row['real_ref_sharpness']:.3f} | "
            f"{row['candidate_blockiness']:.3f} | {row['real_ref_blockiness']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Score sources:",
            "",
        ]
    )
    if score_sources:
        lines.extend(f"- `{source}`" for source in score_sources)
    else:
        lines.append(f"- `{args.run_dir / 'candidate_quality_scores.csv'}`")
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            f"- `figure_best{args.top_n}_generated_perturbation_images.png/svg/pdf/tiff`",
            "- `figure_all44_generated_image_quality_ranking.png/svg/pdf/tiff`",
            f"- `best{args.top_n}_generated_perturbation_images.csv`",
            "- `all44_best_generated_image_quality.csv`",
            "",
            "Selection note:",
            "",
            "This ranking is based on image morphology/display statistics, not RNA Pearson. It is intended to",
            "select the clearest and most normal-looking generated perturbation images for visual showcase.",
        ]
    )
    (args.output_dir / f"best{args.top_n}_generated_perturbation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT.parent)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "comparison_outputs/full44_imageft_morphology_candidates",
    )
    p.add_argument(
        "--score-csv",
        type=Path,
        nargs="*",
        default=None,
        help="Optional candidate_quality_scores.csv files to merge before selection.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "comparison_outputs/best3_generated_perturbation_images",
    )
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--keep-per-drug", type=int, default=3)
    args = p.parse_args()
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

    if args.score_csv:
        frames = []
        for csv_path in args.score_csv:
            csv_path = resolve_path(str(csv_path), args.root)
            frame = pd.read_csv(csv_path)
            frame["source_csv"] = str(csv_path)
            frames.append(frame)
        scores = pd.concat(frames, ignore_index=True)
    else:
        scores = pd.read_csv(args.run_dir / "candidate_quality_scores.csv")
        scores["source_csv"] = str(args.run_dir / "candidate_quality_scores.csv")
    args.scores = scores
    best = add_selection_metrics(best_candidate_per_drug(scores)).sort_values("score", ascending=False).reset_index(drop=True)
    selected = best.head(args.top_n).copy()
    best.to_csv(args.output_dir / "all44_best_generated_image_quality.csv", index=False)
    selected.to_csv(args.output_dir / f"best{args.top_n}_generated_perturbation_images.csv", index=False)
    plot_selected_plate(selected, args)
    plot_all_drug_ranking(best, selected, args)
    write_report(best, selected, args)
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
