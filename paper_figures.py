#!/usr/bin/env python3
"""
IEEE-sized figures for the paper:
  fig_arch_compact  full-width architecture figure (7.16 in wide), same content as arch_figure.py
  fig_e1e2          two-panel figure: time to result versus uplink (a) and versus outage rate (b)

Needs, in the same folder: edge_experiments.py, arch_figure.py (your architecture script) and the
CSV files from an edge_experiments.py run (e1_bandwidth.csv, e2_outage.csv).

Usage:
    python paper_figures.py --results results_typical_112MB --out paper_figures
"""
import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import arch_figure as af
import edge_experiments as ee


def save(fig, out: Path, name: str, dpi: int):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_rows(path: Path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def fig_arch_compact():
    n = len(af.LAYERS)
    row_h, gap, leg_h, title_w = 0.68, 0.17, 0.42, 1.38
    W = 7.16
    H = n * row_h + (n - 1) * gap + leg_h
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    box_w, box_gap, box_h = 1.36, 0.06, 0.50
    x0 = title_w + 0.08
    for i, (title, comps) in enumerate(af.LAYERS):
        top = H - i * (row_h + gap)
        y0 = top - row_h
        ax.add_patch(FancyBboxPatch((0.02, y0), W - 0.04, row_h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=af.LAYER_FILL, ec=af.LAYER_EDGE, lw=0.7, zorder=1))
        ax.text(0.12, y0 + row_h / 2, textwrap.fill(title, 17), ha="left", va="center",
                fontsize=7.2, fontweight="bold", color="#2C2C2A", zorder=2)
        for j, (name, detail, status) in enumerate(comps):
            fill, edge, txt, ls = af.STYLE[status]
            x = x0 + j * (box_w + box_gap)
            y = y0 + (row_h - box_h) / 2
            ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=fill, ec=edge, lw=0.9, ls=ls, zorder=3))
            ax.text(x + box_w / 2, y + box_h * 0.68, name, ha="center", va="center",
                    fontsize=6.7, fontweight="bold", color=txt, zorder=4)
            ax.text(x + box_w / 2, y + box_h * 0.27, detail, ha="center", va="center",
                    fontsize=5.6, color=txt, zorder=4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((W / 2 + 0.6, y0), (W / 2 + 0.6, y0 - gap),
                                         arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                         color="#5F5E5A", zorder=2, shrinkA=0, shrinkB=0))
    for status, lx in (("measured", 0.12), ("sim", 1.95), ("design", 5.0)):
        fill, edge, _, ls = af.STYLE[status]
        ax.add_patch(FancyBboxPatch((lx, 0.09), 0.30, 0.19,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fill, ec=edge, lw=0.9, ls=ls))
        ax.text(lx + 0.38, 0.185, af.STATUS_LABEL[status], ha="left", va="center",
                fontsize=6.8, color="#2C2C2A")
    return fig


def fig_e1e2(res: Path):
    r1 = read_rows(res / "e1_bandwidth.csv")
    r2 = read_rows(res / "e2_outage.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    series = (("e50", ee.TEAL, "-", "o", "Edge, median"), ("e95", ee.TEAL, ":", "o", "Edge, 95th pct"),
              ("c50", ee.PURPLE, "--", "s", "Cloud, median"), ("c95", ee.PURPLE, "-.", "s", "Cloud, 95th pct"))
    x1 = [r["wan_mbps"] for r in r1]
    x2 = [100 * r["outage_prob"] for r in r2]
    for key, c, ls, mk, lab in series:
        ee.band(axs[0], x1, [{f"{key}_mean": r[f"{key}_mean"], f"{key}_ci": r[f"{key}_ci"]} for r in r1],
                key, c, ls, mk, lab)
        ee.band(axs[1], x2, [{f"{key}_mean": r[f"{key}_mean"], f"{key}_ci": r[f"{key}_ci"]} for r in r2],
                key, c, ls, mk, lab)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[1].set_yscale("log")
    axs[0].set_xlabel("Uplink bandwidth (Mbps)"); axs[0].set_ylabel("Time to result (s)")
    axs[1].set_xlabel("WAN outage probability per visit (%) at 50 Mbps")
    axs[0].set_title("(a) Versus uplink bandwidth", fontsize=8)
    axs[1].set_title("(b) Versus outage probability", fontsize=8)
    axs[0].legend(frameon=False, loc="upper right")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results_typical_112MB")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ee.style()
    save(fig_arch_compact(), out, "fig_arch_compact", a.dpi)
    save(fig_e1e2(Path(a.results)), out, "fig_e1e2", a.dpi)
    print("saved fig_arch_compact and fig_e1e2 in", out)


if __name__ == "__main__":
    main()
