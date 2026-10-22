#!/usr/bin/env python3
"""
Figure: edge inference time versus CT study size (measured), and the resulting simulated
time to result for edge and cloud at the measured uplink.

Edit MEASURED with your own timings (study MB, mean s, min s, max s). Panel (b) uses the
same latency model as edge_experiments.py (keep this file in the same folder).

Usage:
    python fig_size_scaling.py
    python fig_size_scaling.py --wan-mbps 21.2 --rtt-ms 49.8 --cloud-infer-s 4 --out figures
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import edge_experiments as ee

# study MB, mean s, min s, max s  (full path DICOM read -> risk score, warm cache, 3 runs)
MEASURED = [(30.0, 3.39, 2.91, 4.03), (112.0, 6.48, 5.70, 7.98),
            (514.9, 11.22, 6.72, 19.09), (1610.4, 50.08, 48.63, 52.29)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wan-mbps", type=float, default=21.2)
    ap.add_argument("--rtt-ms", type=float, default=49.8)
    ap.add_argument("--cloud-infer-s", type=float, default=4.0)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    ee.style()
    rng = np.random.default_rng(1)
    mb = np.array([m[0] for m in MEASURED]); mean = np.array([m[1] for m in MEASURED])
    lo = mean - np.array([m[2] for m in MEASURED]); hi = np.array([m[3] for m in MEASURED]) - mean

    res, rows = [], []
    for size, t, tmin, tmax in MEASURED:
        p = ee.P(study_mb=size, wan_mbps=a.wan_mbps, rtt_ms=a.rtt_ms,
                 edge_infer_s=t, cloud_infer_s=a.cloud_infer_s)
        e, c = ee.sample_times(p, 100000, rng)
        be_main = ee.breakeven_bw(p, rng)
        be_fast = ee.breakeven_bw(replace(p, cloud_infer_s=0.5), rng)
        res.append((np.median(e), np.percentile(e, 95), np.median(c), np.percentile(c, 95)))
        rows.append({"study_mb": size, "edge_infer_mean_s": t, "edge_infer_min_s": tmin,
                     "edge_infer_max_s": tmax,
                     "edge_p50_s": float(round(res[-1][0], 1)), "edge_p95_s": float(round(res[-1][1], 1)),
                     "cloud_p50_s": float(round(res[-1][2], 1)), "cloud_p95_s": float(round(res[-1][3], 1)),
                     "breakeven_mbps_cloud_infer_%gs" % a.cloud_infer_s:
                         "none below 2000" if be_main == float("inf") else round(be_main),
                     "breakeven_mbps_cloud_infer_0.5s":
                         "none below 2000" if be_fast == float("inf") else round(be_fast)})
    res = np.array(res)

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axs[0]
    ax.errorbar(mb, mean, yerr=[lo, hi], color=ee.TEAL, marker="o", ms=4, lw=1.2, capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Study size (MB)"); ax.set_ylabel("Edge inference time (s)")
    ax.set_title("(a) Measured, CPU edge node", fontsize=8)
    ax = axs[1]
    ax.plot(mb, res[:, 0], color=ee.TEAL, ls="-", marker="o", ms=4, lw=1.2, label="Edge, median")
    ax.plot(mb, res[:, 1], color=ee.TEAL, ls=":", marker="o", ms=4, lw=1.2, label="Edge, 95th pct")
    ax.plot(mb, res[:, 2], color=ee.PURPLE, ls="--", marker="s", ms=4, lw=1.2, label="Cloud, median")
    ax.plot(mb, res[:, 3], color=ee.PURPLE, ls="-.", marker="s", ms=4, lw=1.2, label="Cloud, 95th pct")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Study size (MB)"); ax.set_ylabel("Time to result (s)")
    ax.set_title(f"(b) Simulated, {a.wan_mbps:g} Mbps uplink", fontsize=8)
    ax.legend(frameon=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_size_scaling.{ext}", dpi=a.dpi, bbox_inches="tight", facecolor="white")
    with open(out / "size_scaling_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", out / "fig_size_scaling.png", "and", out / "size_scaling_table.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
