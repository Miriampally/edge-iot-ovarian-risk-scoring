#!/usr/bin/env python3
"""
Experiment suite for the software-defined edge IIoT architecture (ovarian cancer CT visits).

Five experiments compare EDGE inference (study stays in the clinic) with CLOUD inference
(full CT study uploaded over the WAN). Time to result = scan complete -> result ready for
the on-site clinician.

  E1  time to result vs uplink bandwidth            (Monte Carlo, CI over repeats)
  E2  time to result vs WAN outage probability      (Monte Carlo, CI over repeats)
  E3  response time vs visit arrival rate           (queueing: edge GPU queue vs shared uplink queue)
  E4  uplink volume saved vs federated-round size   (analytic)
  E5  break-even map: cloud minus edge median time over uplink speed x edge inference time

IMPORTANT: every number here comes from a parameterised MODEL, not from measurements.
Set the PARAMETERS block (class P) from your own benchmarks (measured edge/cloud inference
time, real uplink speed and round-trip time, real study size) before quoting results.
Model assumptions are listed in RESULTS.md after each run.

Usage
    python edge_experiments.py                  # full run, ~1-2 minutes
    python edge_experiments.py --quick          # fast smoke test
    python edge_experiments.py --out results --seed 7 --study-mb 150 --edge-infer-s 20

Requires: numpy, matplotlib
Outputs (in --out): e1..e5 CSV files, figures (PNG + PDF, single-column IEEE width),
                    RESULTS.md (tables and key numbers to paste into the paper)
"""
from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

TEAL, PURPLE, GRAY = "#0F6E56", "#534AB7", "#5F5E5A"


# --------------------------------------------------------------------------
# Parameters (edit these from your own benchmarks)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class P:
    study_mb: float = 200.0          # size of one CT study
    lan_mbps: float = 1000.0         # scanner -> edge node
    wan_mbps: float = 100.0          # clinic -> cloud uplink
    rtt_ms: float = 40.0             # clinic <-> cloud round trip
    outage_prob: float = 0.05        # chance the WAN is down for a visit
    outage_s: tuple = (60.0, 600.0)  # outage length range (s)
    edge_infer_s: float = 12.0       # mean inference time on the edge node
    cloud_infer_s: float = 4.0       # mean inference time on a cloud GPU
    overhead_s: float = 0.05         # policy check + de-identification (assumed)
    cv: float = 0.15                 # coefficient of variation of inference times
    model_mb: float = 100.0          # one federated weight update
    visits_per_round: int = 50       # visits between federated updates
    result_kb: float = 2.0           # size of the result message


# --------------------------------------------------------------------------
# Core latency model (vectorised)
# --------------------------------------------------------------------------
def sample_times(p: P, n: int, rng: np.random.Generator):
    lan = (p.study_mb * 8 / p.lan_mbps + 0.001) * rng.uniform(0.9, 1.4, n)
    inf_e = np.maximum(0.01, rng.normal(p.edge_infer_s, p.edge_infer_s * p.cv, n))
    inf_c = np.maximum(0.01, rng.normal(p.cloud_infer_s, p.cloud_infer_s * p.cv, n))
    wan = (p.study_mb * 8 / p.wan_mbps + p.rtt_ms / 1000) * rng.uniform(0.9, 1.4, n)
    wait = np.where(rng.random(n) < p.outage_prob, rng.uniform(*p.outage_s, n), 0.0)
    ret = p.rtt_ms / 1000 * rng.uniform(0.9, 1.4, n)
    edge = p.overhead_s + lan + inf_e
    cloud = p.overhead_s + lan + wan + wait + inf_c + ret
    return edge, cloud


def ci(a) -> tuple[float, float]:
    a = np.asarray(a, float)
    h = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(h)


def mc_stats(p: P, rng, n: int, reps: int) -> dict:
    r = {k: [] for k in ("e50", "e95", "c50", "c95")}
    for _ in range(reps):
        e, c = sample_times(p, n, rng)
        r["e50"].append(np.median(e)); r["e95"].append(np.percentile(e, 95))
        r["c50"].append(np.median(c)); r["c95"].append(np.percentile(c, 95))
    return {k: ci(v) for k, v in r.items()}


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def e1_bandwidth(p, rng, n, reps):
    rows = []
    for bw in [5, 10, 20, 50, 100, 200, 500, 1000]:
        s = mc_stats(replace(p, wan_mbps=bw), rng, n, reps)
        rows.append({"wan_mbps": bw, **{f"{k}_{t}": round(s[k][i], 3)
                     for k in s for i, t in enumerate(("mean", "ci"))}})
    return rows


def e2_outage(p, rng, n, reps):
    rows = []
    for po in [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        s = mc_stats(replace(p, wan_mbps=50, outage_prob=po), rng, n, reps)
        rows.append({"outage_prob": po, **{f"{k}_{t}": round(s[k][i], 3)
                     for k in s for i, t in enumerate(("mean", "ci"))}})
    return rows


def sim_queue(p: P, lam_per_h: float, n: int, edge_infer_s: float, k_edge: int, rng):
    """Edge: k_edge servers FIFO. Cloud: one shared FIFO uplink, then parallel cloud inference."""
    arr = np.cumsum(rng.exponential(3600 / lam_per_h, n))
    pe = replace(p, edge_infer_s=edge_infer_s, outage_prob=0.0)
    lan = (p.study_mb * 8 / p.lan_mbps + 0.001) * rng.uniform(0.9, 1.4, n)
    inf_e = np.maximum(0.01, rng.normal(pe.edge_infer_s, pe.edge_infer_s * p.cv, n))
    inf_c = np.maximum(0.01, rng.normal(p.cloud_infer_s, p.cloud_infer_s * p.cv, n))
    up = (p.study_mb * 8 / p.wan_mbps) * rng.uniform(0.9, 1.4, n)
    ret = p.rtt_ms / 1000 * rng.uniform(0.9, 1.4, n)
    free = [0.0] * k_edge
    heapq.heapify(free)
    edge = np.empty(n); cloud = np.empty(n); up_free = 0.0
    for i, a in enumerate(arr):
        s = heapq.heappop(free)
        end = max(a, s) + inf_e[i]
        heapq.heappush(free, end)
        edge[i] = p.overhead_s + lan[i] + (end - a)
        start = max(a, up_free)
        up_free = start + up[i]
        cloud[i] = p.overhead_s + lan[i] + (up_free - a) + p.rtt_ms / 1000 + inf_c[i] + ret[i]
    w = n // 10  # discard warm-up
    return edge[w:], cloud[w:]


def e3_load(p, rng, n, reps, slow_edge_s):
    rows = []
    for lam in [20, 50, 100, 200, 300, 400, 500, 600]:
        acc = {k: [] for k in ("e50", "e95", "s50", "s95", "c50", "c95")}
        for _ in range(reps):
            e, c = sim_queue(p, lam, n, p.edge_infer_s, 1, rng)
            s, _ = sim_queue(p, lam, n, slow_edge_s, 1, rng)
            acc["e50"].append(np.median(e)); acc["e95"].append(np.percentile(e, 95))
            acc["s50"].append(np.median(s)); acc["s95"].append(np.percentile(s, 95))
            acc["c50"].append(np.median(c)); acc["c95"].append(np.percentile(c, 95))
        rows.append({"arrivals_per_hour": lam, **{f"{k}_mean": round(float(np.mean(v)), 2)
                     for k, v in acc.items()}})
    return rows


def e4_uplink(p):
    rows = []
    for model in [25, 100, 400]:
        for vpr in [10, 25, 50, 100, 250, 500]:
            edge_mb = p.result_kb / 1000 + model / vpr
            rows.append({"model_mb": model, "visits_per_round": vpr,
                         "edge_mb_per_visit": round(edge_mb, 3),
                         "cloud_mb_per_visit": p.study_mb,
                         "saved_pct": round(100 * (1 - edge_mb / p.study_mb), 2)})
    return rows


def e5_breakeven(p, rng):
    bws = [5, 10, 20, 50, 100, 200, 500, 1000]
    edges = [3.4, 6.5, 11.2, 30, 50.1, 120, 300]   # first, second, third and fifth are measured
    grid = np.zeros((len(edges), len(bws)))
    for i, ei in enumerate(edges):
        for j, bw in enumerate(bws):
            e, c = sample_times(replace(p, wan_mbps=bw, edge_infer_s=ei), 3000, rng)
            grid[i, j] = np.median(c) - np.median(e)
    return bws, edges, grid


def breakeven_bw(p, rng) -> float:
    """Uplink speed (Mbps) above which cloud median time <= edge median time."""
    bws = np.geomspace(5, 2000, 60)
    d = []
    for bw in bws:
        e, c = sample_times(replace(p, wan_mbps=float(bw)), 20000, rng)
        d.append(np.median(c) - np.median(e))
    d = np.array(d)
    idx = np.where(d <= 0)[0]
    if len(idx) == 0:
        return float("inf")
    k = idx[0]
    if k == 0:
        return float(bws[0])
    f = d[k - 1] / (d[k - 1] - d[k])
    return float(np.exp(np.log(bws[k - 1]) + f * (np.log(bws[k]) - np.log(bws[k - 1]))))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def style():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def band(ax, x, rows, key, color, ls, marker, label):
    m = np.array([r[f"{key}_mean"] for r in rows]); h = np.array([r[f"{key}_ci"] for r in rows])
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=3, lw=1.2, label=label)
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.15, lw=0)


def fig_e1(rows, out):
    x = [r["wan_mbps"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    band(ax, x, rows, "e50", TEAL, "-", "o", "Edge, median")
    band(ax, x, rows, "e95", TEAL, ":", "o", "Edge, 95th pct")
    band(ax, x, rows, "c50", PURPLE, "--", "s", "Cloud, median")
    band(ax, x, rows, "c95", PURPLE, "-.", "s", "Cloud, 95th pct")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Uplink bandwidth (Mbps)"); ax.set_ylabel("Time to result (s)")
    ax.legend(frameon=False)
    save(fig, out, "fig_e1_bandwidth")


def fig_e2(rows, out):
    x = [100 * r["outage_prob"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    band(ax, x, rows, "e50", TEAL, "-", "o", "Edge, median")
    band(ax, x, rows, "e95", TEAL, ":", "o", "Edge, 95th pct")
    band(ax, x, rows, "c50", PURPLE, "--", "s", "Cloud, median")
    band(ax, x, rows, "c95", PURPLE, "-.", "s", "Cloud, 95th pct")
    ax.set_yscale("log")
    ax.set_xlabel("WAN outage probability per visit (%)"); ax.set_ylabel("Time to result (s)")
    ax.legend(frameon=False)
    save(fig, out, "fig_e2_outage")


def fig_e3(rows, out, slow, caps, fast):
    x = [r["arrivals_per_hour"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, q, title in ((axs[0], "50", "(a) Median"), (axs[1], "95", "(b) 95th percentile")):
        for key, c, ls, mk, lab in (("e", TEAL, "-", "o", f"Edge, {fast:g} s inference"),
                                    ("s", GRAY, "--", "^", f"Edge, {slow:g} s inference"),
                                    ("c", PURPLE, "-.", "s", "Cloud (shared uplink)")):
            ax.plot(x, [r[f"{key}{q}_mean"] for r in rows], color=c, ls=ls, marker=mk,
                    ms=3, lw=1.2, label=lab)
        for cap, c in zip(caps, (TEAL, GRAY, PURPLE)):  # saturation points
            ax.axvline(cap, color=c, lw=0.7, ls=(0, (1, 2)), alpha=0.8)
        ax.set_yscale("log"); ax.set_ylim(3, 3000); ax.set_title(title, fontsize=8)
        ax.set_xlabel("Visit arrivals per hour")
    axs[0].set_ylabel("Response time (s)")
    axs[0].legend(frameon=False, loc="upper left")
    # Caption note for the paper: dotted lines mark each path's capacity; beyond them the queue
    # grows without bound (y-axis clipped at 3000 s).
    save(fig, out, "fig_e3_load")


def fig_e4(rows, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for model, ls, mk in ((25, "-", "o"), (100, "--", "s"), (400, ":", "^")):
        r = [x for x in rows if x["model_mb"] == model]
        ax.plot([x["visits_per_round"] for x in r], [x["saved_pct"] for x in r],
                color=TEAL, ls=ls, marker=mk, ms=3, lw=1.2, label=f"{model} MB update")
    ax.set_xscale("log")
    ax.set_xlabel("Visits per federated round"); ax.set_ylabel("Uplink saved vs. cloud (%)")
    ax.legend(frameon=False)
    save(fig, out, "fig_e4_uplink")


def fig_e5(bws, edges, grid, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    lim = max(abs(grid.min()), abs(grid.max()))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu",
                   norm=SymLogNorm(linthresh=10, vmin=-lim, vmax=lim, base=10))
    ax.set_xticks(range(len(bws)), [str(b) for b in bws])
    ax.set_yticks(range(len(edges)), [str(e) for e in edges])
    for i in range(len(edges)):
        for j in range(len(bws)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(grid[i, j]) > 0.55 * lim else "black")
    ax.contour(grid, levels=[0], colors="k", linewidths=1.2)
    ax.grid(False)
    ax.set_xlabel("Uplink bandwidth (Mbps)"); ax.set_ylabel("Edge inference time (s)")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Cloud minus edge median (s)\n(positive: edge faster)", fontsize=6.5)
    save(fig, out, "fig_e5_breakeven")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def pm(r, k):
    return f"{r[f'{k}_mean']:.1f} ± {r[f'{k}_ci']:.1f}"


def write_results(out, p, args, e1, e2, e3, e4, bw_be, slow):
    L = ["# Simulation results (modeled)\n",
         "All values come from a parameterised model, not measurements. "
         "Mean ± 95% CI over repeats where shown.\n",
         "## Parameters\n", "| Parameter | Value |", "| --- | --- |"]
    for k, v in p.__dict__.items():
        L.append(f"| {k} | {v} |")
    L += [f"| seed | {args.seed} |", f"| visits per repeat / repeats | {args.n} / {args.reps} |", ""]
    L += [f"**Break-even uplink:** with these parameters the cloud median time matches the edge "
          f"median at about {bw_be:.0f} Mbps; below that the edge is faster.\n",
          "## E1: time to result vs uplink bandwidth (s)\n",
          "| Uplink (Mbps) | Edge p50 | Edge p95 | Cloud p50 | Cloud p95 |", "| --- | --- | --- | --- | --- |"]
    for r in e1:
        L.append(f"| {r['wan_mbps']} | {pm(r,'e50')} | {pm(r,'e95')} | {pm(r,'c50')} | {pm(r,'c95')} |")
    L += ["", "## E2: time to result vs outage probability at 50 Mbps (s)\n",
          "| Outage (%) | Edge p50 | Edge p95 | Cloud p50 | Cloud p95 |", "| --- | --- | --- | --- | --- |"]
    for r in e2:
        L.append(f"| {100*r['outage_prob']:.0f} | {pm(r,'e50')} | {pm(r,'e95')} | {pm(r,'c50')} | {pm(r,'c95')} |")
    L += ["", f"## E3: response time under load (s; one edge GPU; {p.wan_mbps:g} Mbps shared uplink)\n",
          f"| Arrivals/h | Edge {p.edge_infer_s:g} s p50 | Edge {slow:g} s p50 | Cloud p50 | Edge {p.edge_infer_s:g} s p95 | Edge {slow:g} s p95 | Cloud p95 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in e3:
        L.append(f"| {r['arrivals_per_hour']} | {r['e50_mean']:.1f} | {r['s50_mean']:.1f} | {r['c50_mean']:.1f} "
                 f"| {r['e95_mean']:.1f} | {r['s95_mean']:.1f} | {r['c95_mean']:.1f} |")
    cap_e = 3600 / p.edge_infer_s; cap_s = 3600 / slow
    cap_c = 3600 / (p.study_mb * 8 / p.wan_mbps * 1.15)
    L += ["", f"Approximate capacity (visits/h): edge {p.edge_infer_s:g} s = {cap_e:.0f}, "
          f"edge {slow:g} s = {cap_s:.0f}, shared uplink = {cap_c:.0f}. "
          "Above capacity the queue grows without bound, so those response times depend on run length.\n",
          "## E4: uplink per visit\n",
          "| Model (MB) | Visits/round | Edge (MB/visit) | Cloud (MB/visit) | Saved (%) |", "| --- | --- | --- | --- | --- |"]
    for r in e4:
        L.append(f"| {r['model_mb']} | {r['visits_per_round']} | {r['edge_mb_per_visit']} | "
                 f"{r['cloud_mb_per_visit']:.0f} | {r['saved_pct']} |")
    L += ["", "## Model assumptions\n",
          "- Time to result ends when the result is available to the on-site clinician; remote delivery is queued.",
          "- De-identification runs at the edge in both deployments, so only de-identified data leave the clinic.",
          "- Transfer times are size / bandwidth times a uniform factor in [0.9, 1.4] plus round-trip time.",
          "- Inference times are normal with the stated coefficient of variation.",
          "- Outages add a uniform wait to the cloud path only. E3 excludes outages to isolate the load effect.",
          "- E3: edge is a FIFO queue with one GPU; cloud shares one FIFO uplink and has unlimited cloud GPUs.",
          "- E4 counts uplink only and ignores the downlink used to roll out global models."]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="fewer repeats for a fast test")
    ap.add_argument("--study-mb", type=float, default=P.study_mb)
    ap.add_argument("--wan-mbps", type=float, default=P.wan_mbps)
    ap.add_argument("--rtt-ms", type=float, default=P.rtt_ms)
    ap.add_argument("--outage-prob", type=float, default=P.outage_prob)
    ap.add_argument("--edge-infer-s", type=float, default=P.edge_infer_s)
    ap.add_argument("--cloud-infer-s", type=float, default=P.cloud_infer_s)
    ap.add_argument("--slow-edge-s", type=float, default=30.0,
                    help="slower edge hardware shown in E3 for contrast")
    a = ap.parse_args()
    a.n, a.reps = (300, 5) if a.quick else (1000, 20)
    q_n, q_reps = (800, 3) if a.quick else (3000, 8)

    p = P(study_mb=a.study_mb, wan_mbps=a.wan_mbps, rtt_ms=a.rtt_ms,
          outage_prob=a.outage_prob, edge_infer_s=a.edge_infer_s, cloud_infer_s=a.cloud_infer_s)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    style()

    print("E1 bandwidth sweep ...");  e1 = e1_bandwidth(p, rng, a.n, a.reps)
    print("E2 outage sweep ...");     e2 = e2_outage(p, rng, a.n, a.reps)
    print("E3 load / queueing ...");  e3 = e3_load(p, rng, q_n, q_reps, a.slow_edge_s)
    print("E4 uplink volume ...");    e4 = e4_uplink(p)
    print("E5 break-even map ...");   bws, edges, grid = e5_breakeven(p, rng)
    bw_be = breakeven_bw(p, rng)

    for name, rows in (("e1_bandwidth", e1), ("e2_outage", e2), ("e3_load", e3), ("e4_uplink", e4)):
        write_csv(out / f"{name}.csv", rows)
    caps = (3600 / p.edge_infer_s, 3600 / a.slow_edge_s,
            3600 / (p.study_mb * 8 / p.wan_mbps * 1.15))
    fig_e1(e1, out); fig_e2(e2, out); fig_e3(e3, out, a.slow_edge_s, caps, p.edge_infer_s); fig_e4(e4, out)
    fig_e5(bws, edges, grid, out)
    write_results(out, p, a, e1, e2, e3, e4, bw_be, a.slow_edge_s)

    print(f"\nBreak-even uplink (median): ~{bw_be:.0f} Mbps (edge faster below this)")
    r = next(x for x in e1 if x["wan_mbps"] == 20)
    print(f"At 20 Mbps: edge p50 {r['e50_mean']:.1f}s / p95 {r['e95_mean']:.1f}s;"
          f" cloud p50 {r['c50_mean']:.1f}s / p95 {r['c95_mean']:.1f}s")
    print(f"Files written to {out}/ (CSV, PNG/PDF figures, RESULTS.md)")


if __name__ == "__main__":
    main()
