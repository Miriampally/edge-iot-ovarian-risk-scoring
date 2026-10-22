#!/usr/bin/env python3
"""
Layered architecture figure for the software-defined edge IoT risk-scoring system
(Figure 1). Edit LAYERS below to change the content; the drawing code does not
need to be touched.

Each component has a status that decides its style:
    measured  - timed on real hardware (edge inference)
    sim       - implemented in the visit simulator (policy check, de-identification,
                audit log, federated update modelled as bytes)
    design    - part of the design only, not implemented (dashed border)

Usage:
    python arch_figure.py                      # PNG + PDF + SVG in ./figures
    python arch_figure.py --out out --dpi 600

Requires: matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# status -> (fill, border, text, linestyle)
STYLE = {
    "measured": ("#E1F5EE", "#0F6E56", "#04342C", "-"),
    "sim": ("#EEEDFE", "#534AB7", "#26215C", "-"),
    "design": ("#FFFFFF", "#5F5E5A", "#2C2C2A", (0, (4, 2.5))),
}
STATUS_LABEL = {
    "measured": "Measured on hardware",
    "sim": "Modelled or implemented in simulator",
    "design": "Design element, not implemented",
}

# (layer title, [(name, detail, status), ...])
LAYERS = [
    ("Patient and device layer", [
        ("CT scanner", "DICOM study, LAN transfer", "sim"),
        ("Intake and consent", "Consent flag", "sim"),
    ]),
    ("Edge layer (clinic)", [
        ("Ingest, de-identify", "Header scrub, pseudonym", "sim"),
        ("Edge inference", "Embedding, risk score", "measured"),
        ("Local cache", "Encrypted store", "design"),
    ]),
    ("Software-defined control plane", [
        ("Policy and token", "Consent, scope, expiry", "sim"),
        ("Audit log", "Hash-chained", "sim"),
        ("SDN slicing, QoS", "Traffic priority", "design"),
        ("Orchestrator", "Model rollout", "design"),
    ]),
    ("Cloud and core", [
        ("Federated hub", "Weights-only updates", "sim"),
        ("Cloud inference", "Baseline for comparison", "sim"),
        ("EHR and PACS", "Records, archive", "design"),
    ]),
    ("Clinician layer", [
        ("Review workstation", "Score, uncertainty, XAI", "sim"),
        ("Report and alerts", "Result to clinic", "sim"),
    ]),
]

LAYER_FILL, LAYER_EDGE = "#F1EFE8", "#5F5E5A"
ROW_W, ROW_H, GAP = 10.0, 1.9, 0.75
BOX_H, BOX_PAD = 1.0, 0.35


def draw(font: float):
    plt.rcParams.update({"font.size": font, "font.family": "DejaVu Sans"})
    n = len(LAYERS)
    height = n * ROW_H + (n - 1) * GAP + 0.9
    fig, ax = plt.subplots(figsize=(8.6, height * 0.78))
    ax.set_xlim(0, ROW_W)
    ax.set_ylim(0.0, height - 0.1)
    ax.axis("off")

    tops = []
    for i, (title, comps) in enumerate(LAYERS):
        top = height - 0.1 - i * (ROW_H + GAP)
        y0 = top - ROW_H
        tops.append((top, y0))
        ax.add_patch(FancyBboxPatch((0.05, y0), ROW_W - 0.1, ROW_H,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    fc=LAYER_FILL, ec=LAYER_EDGE, lw=1.0, zorder=1))
        ax.text(0.3, top - 0.3, title, ha="left", va="center", fontsize=font + 1,
                color="#2C2C2A", zorder=2)
        k = len(comps)
        inner = ROW_W - 2 * 0.3
        w = (inner - (k - 1) * BOX_PAD) / k
        for j, (name, detail, status) in enumerate(comps):
            fill, edge, txt, ls = STYLE[status]
            x = 0.3 + j * (w + BOX_PAD)
            y = y0 + 0.2
            ax.add_patch(FancyBboxPatch((x, y), w, BOX_H,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=fill, ec=edge, lw=1.2, ls=ls, zorder=3))
            ax.text(x + w / 2, y + BOX_H * 0.64, name, ha="center", va="center",
                    fontsize=font, fontweight="bold", color=txt, zorder=4)
            ax.text(x + w / 2, y + BOX_H * 0.30, detail, ha="center", va="center",
                    fontsize=font - 1.5, color=txt, zorder=4)

    for i in range(n - 1):
        y_from = tops[i][1]
        y_to = tops[i + 1][0]
        ax.add_patch(FancyArrowPatch((ROW_W / 2, y_from), (ROW_W / 2, y_to),
                                     arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                     color="#5F5E5A", zorder=2, shrinkA=1, shrinkB=1))

    # legend
    lx = 0.3
    ly = 0.15
    for status in ("measured", "sim", "design"):
        fill, edge, _, ls = STYLE[status]
        ax.add_patch(FancyBboxPatch((lx, ly), 0.45, 0.32,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fill, ec=edge, lw=1.2, ls=ls))
        ax.text(lx + 0.6, ly + 0.16, STATUS_LABEL[status], ha="left", va="center",
                fontsize=font - 1.5, color="#2C2C2A")
        lx += 3.3
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="architecture")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-size", type=float, default=9.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw(a.font_size)
    for ext in ("png", "pdf", "svg"):
        p = out / f"{a.name}.{ext}"
        fig.savefig(p, dpi=a.dpi, bbox_inches="tight", facecolor="white")
        print("saved", p)


if __name__ == "__main__":
    main()
