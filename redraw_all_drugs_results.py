"""
Redraw all-drug RNA comparison results from the completed comparison table.

This produces a compact, readable all-44-drug figure focused on:
1. RNA Pearson for CPFLOW combined versus original MultiVCDiff.
2. Per-drug delta Pearson (CPFLOW - MultiVCDiff).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def save_pub(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "comparison_outputs/full_44drugs_cpflow50_mvc50",
    )
    parser.add_argument("--sort-by", choices=["cpflow", "delta"], default="cpflow")
    args = parser.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

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

    base = args.comparison_dir
    deltas = pd.read_csv(base / "per_drug_deltas.csv")
    required = {
        "drug",
        "pearson_CPFLOW_combined",
        "pearson_MultiVCDiff_original",
        "delta_pearson_cpflow_minus_multivcdiff",
    }
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"Missing required columns in per_drug_deltas.csv: {sorted(missing)}")

    if args.sort_by == "delta":
        plot_df = deltas.sort_values(
            ["delta_pearson_cpflow_minus_multivcdiff", "pearson_CPFLOW_combined"],
            ascending=[True, True],
        ).reset_index(drop=True)
        suffix = "sorted_by_delta"
    else:
        plot_df = deltas.sort_values(
            ["pearson_CPFLOW_combined", "delta_pearson_cpflow_minus_multivcdiff"],
            ascending=[True, True],
        ).reset_index(drop=True)
        suffix = "sorted_by_cpflow"

    plot_df.to_csv(base / f"all_drugs_redraw_source_{suffix}.csv", index=False)

    x = np.arange(plot_df.shape[0])
    cp = plot_df["pearson_CPFLOW_combined"].to_numpy()
    mv = plot_df["pearson_MultiVCDiff_original"].to_numpy()
    delta = plot_df["delta_pearson_cpflow_minus_multivcdiff"].to_numpy()
    drugs = plot_df["drug"].tolist()

    cp_color = "#3B7EA1"
    mv_color = "#B45F4D"
    delta_color = "#4D7C8A"
    neutral = "#B8B8B8"

    fig = plt.figure(figsize=(12.0, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    for xi, y0, y1 in zip(x, mv, cp):
        ax0.plot([xi, xi], [y0, y1], color=neutral, lw=0.9, zorder=1)
    ax0.scatter(x, mv, color=mv_color, s=15, label="MultiVCDiff original", zorder=3)
    ax0.scatter(x, cp, color=cp_color, s=18, label="CPFLOW combined", zorder=4)
    ax0.axhline(plot_df["pearson_CPFLOW_combined"].mean(), color=cp_color, lw=0.8, ls="--", alpha=0.65)
    ax0.axhline(plot_df["pearson_MultiVCDiff_original"].mean(), color=mv_color, lw=0.8, ls="--", alpha=0.65)
    ax0.set_ylim(0.2, 1.02)
    ax0.set_ylabel("RNA Pearson")
    ax0.set_title("Per-drug RNA fidelity", fontsize=8)
    ax0.legend(loc="lower left", fontsize=6)
    ax0.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax0.tick_params(axis="x", labelbottom=False)

    ax1.bar(x, delta, color=delta_color, width=0.68)
    ax1.axhline(0, color="#333333", lw=0.8)
    ax1.axhline(delta.mean(), color=delta_color, lw=0.8, ls="--", alpha=0.65)
    ax1.set_ylabel("Delta Pearson\n(CPFLOW - MultiVCDiff)")
    ax1.set_title("Per-drug improvement", fontsize=8)
    ax1.set_ylim(0, max(0.75, delta.max() + 0.04))
    ax1.set_xticks(x)
    ax1.set_xticklabels(drugs, rotation=90, ha="center", fontsize=5.2)
    ax1.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax1.margins(x=0.01)

    mean_cp = cp.mean()
    mean_mv = mv.mean()
    mean_delta = delta.mean()
    fig.suptitle(
        f"All 44 drugs: CPFLOW improves RNA Pearson "
        f"({mean_cp:.3f} vs {mean_mv:.3f}; mean delta +{mean_delta:.3f})",
        fontsize=9,
    )

    out = base / f"figure_all_drugs_pearson_delta_{suffix}"
    save_pub(fig, out)
    plt.close(fig)

    report = base / f"all_drugs_redraw_report_{suffix}.md"
    report.write_text(
        "\n".join(
            [
                "# Redrawn all-drug RNA comparison",
                "",
                f"Sort order: `{args.sort_by}`.",
                "",
                "| Metric | CPFLOW combined | MultiVCDiff original | Delta |",
                "|---|---:|---:|---:|",
                f"| Mean RNA Pearson | {mean_cp:.6f} | {mean_mv:.6f} | +{mean_delta:.6f} |",
                f"| Median RNA Pearson | {np.median(cp):.6f} | {np.median(mv):.6f} | +{np.median(delta):.6f} |",
                "",
                "Generated files:",
                "",
                f"- `{out.name}.png`",
                f"- `{out.name}.svg`",
                f"- `{out.name}.pdf`",
                f"- `all_drugs_redraw_source_{suffix}.csv`",
            ]
        )
        + "\n"
    )

    print(f"Wrote {out}.png/.svg/.pdf")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
