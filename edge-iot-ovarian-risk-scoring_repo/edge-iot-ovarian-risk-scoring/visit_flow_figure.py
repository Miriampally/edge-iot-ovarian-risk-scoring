#!/usr/bin/env python3
"""
Swimlane figure of the software-defined edge IIoT data flow for one
CT-based ovarian cancer screening visit (patient -> edge -> doctor -> model update).

Edit STEPS / FLOW / FEEDBACK below to change the figure; the drawing code
does not need to be touched.

Usage:
    python visit_flow_figure.py                       # PNG + PDF + SVG in ./figures
    python visit_flow_figure.py --out out --dpi 600   # custom folder and resolution
    python visit_flow_figure.py --title "Screening visit data flow"

Requires: matplotlib (pip install matplotlib)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------
# 1. Content: lanes, steps, and connections
# --------------------------------------------------------------------------
LANES = [
    "Patient and device",
    "Edge layer",
    "Control plane",
    "Cloud and core",
    "Clinician",
]

# fill / border / text colours per lane (flat palette, readable in print)
GRAY = dict(fill="#F1EFE8", edge="#5F5E5A", text="#2C2C2A")
TEAL = dict(fill="#E1F5EE", edge="#0F6E56", text="#04342C")
PURPLE = dict(fill="#EEEDFE", edge="#534AB7", text="#26215C")

LANE_STYLE = {
    "Patient and device": GRAY,
    "Edge layer": TEAL,
    "Control plane": PURPLE,
    "Cloud and core": PURPLE,
    "Clinician": GRAY,
}


@dataclass(frozen=True)
class Step:
    sid: int          # step number shown in the box
    lane: str         # one of LANES
    col: int          # time position (left to right)
    label: str        # box text, use \n for line breaks (keep lines <= 19 chars)


STEPS = [
    Step(1, "Patient and device", 0, "Consent and\nintake"),
    Step(2, "Patient and device", 1, "CT study to\nedge (LAN)"),
    Step(3, "Control plane", 2, "Policy check,\ntoken"),
    Step(4, "Edge layer", 3, "De-identify,\npseudonym"),
    Step(5, "Edge layer", 4, "Embedding and\nrisk score"),
    Step(6, "Clinician", 5, "Clinician review\nscore, uncertainty"),
    Step(7, "Control plane", 6, "Audit log\n(hash-chained)"),
    Step(8, "Cloud and core", 7, "Weights-only\nmodel update"),
]

# solid arrows: (from step id, to step id)
FLOW = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

# dashed feedback arrows: (from, to, label)
FEEDBACK = []

# --------------------------------------------------------------------------
# 2. Layout constants (data units)
# --------------------------------------------------------------------------
COL_W = 2.8      # distance between columns
BOX_W = 2.25
BOX_H = 0.95
LANE_H = 1.35
LABEL_X = -2.5   # left edge of lane-name column


def lane_y(lane: str) -> float:
    """Vertical centre of a lane (first lane is drawn on top)."""
    return (len(LANES) - 1 - LANES.index(lane)) * LANE_H


def centre(step: Step) -> tuple[float, float]:
    return step.col * COL_W, lane_y(step.lane)


def anchor(step: Step, side: str) -> tuple[float, float]:
    x, y = centre(step)
    return {
        "left": (x - BOX_W / 2, y),
        "right": (x + BOX_W / 2, y),
        "top": (x, y + BOX_H / 2),
        "bottom": (x, y - BOX_H / 2),
    }[side]


# --------------------------------------------------------------------------
# 3. Drawing
# --------------------------------------------------------------------------
def draw(title: str | None, font_size: float):
    plt.rcParams.update({"font.size": font_size, "font.family": "DejaVu Sans"})
    by_id = {s.sid: s for s in STEPS}
    n_cols = max(s.col for s in STEPS) + 1
    x_max = (n_cols - 1) * COL_W + BOX_W / 2 + 0.4

    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(LABEL_X - 0.1, x_max)
    ax.set_ylim(-LANE_H * 0.6, (len(LANES) - 1) * LANE_H + LANE_H * 0.6)
    ax.axis("off")

    # lane bands and lane names
    for i, lane in enumerate(LANES):
        y = lane_y(lane)
        if i % 2 == 0:
            ax.add_patch(
                Rectangle((LABEL_X - 0.1, y - LANE_H / 2), x_max - LABEL_X + 0.1,
                          LANE_H, facecolor="#00000008", edgecolor="none", zorder=0)
            )
        ax.text(LABEL_X, y, lane.replace(" and ", "\nand "), ha="left", va="center",
                fontsize=font_size, fontweight="bold", color=LANE_STYLE[lane]["edge"])

    # solid flow arrows (drawn first so boxes sit on top)
    for a, b in FLOW:
        sa, sb = by_id[a], by_id[b]
        ax.add_patch(
            FancyArrowPatch(anchor(sa, "right"), anchor(sb, "left"),
                            arrowstyle="-|>", mutation_scale=12, lw=1.2,
                            color="#444441", zorder=2, shrinkA=1, shrinkB=1)
        )

    # dashed feedback arrows
    for a, b, text in FEEDBACK:
        sa, sb = by_id[a], by_id[b]
        p1, p2 = anchor(sa, "top"), anchor(sb, "bottom")
        ax.add_patch(
            FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.2,
                            linestyle=(0, (5, 3)), color=PURPLE["edge"], zorder=2,
                            connectionstyle="arc3,rad=0.12", shrinkA=1, shrinkB=1)
        )
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, text, ha="center", va="center", fontsize=font_size - 0.5,
                color=PURPLE["edge"], style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                zorder=4)

    # step boxes
    for s in STEPS:
        st = LANE_STYLE[s.lane]
        x, y = centre(s)
        ax.add_patch(
            FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                           boxstyle="round,pad=0.0,rounding_size=0.12",
                           facecolor=st["fill"], edgecolor=st["edge"], lw=1.1, zorder=3)
        )
        ax.text(x, y, s.label, ha="center", va="center", color=st["text"],
                fontsize=font_size - 0.5, linespacing=1.25, zorder=4)
        ax.text(x - BOX_W / 2 + 0.12, y + BOX_H / 2 - 0.10, str(s.sid),
                ha="left", va="top", fontsize=font_size - 1.5, fontweight="bold",
                color=st["edge"], zorder=4)

    if title:
        ax.set_title(title, fontsize=font_size + 3, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures", help="output folder (default: figures)")
    ap.add_argument("--name", default="visit_data_flow", help="file name without extension")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    ap.add_argument("--font-size", type=float, default=9.0, help="base font size")
    ap.add_argument("--title", default=None, help="optional figure title")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw(args.title, args.font_size)
    for ext in ("png", "pdf", "svg"):
        path = out / f"{args.name}.{ext}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
