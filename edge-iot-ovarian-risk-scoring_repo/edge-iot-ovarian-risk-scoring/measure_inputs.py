#!/usr/bin/env python3
"""
Measure the real inputs that edge_experiments.py needs, so its results stop being assumptions.

It measures, on the machine you run it on:
  1. Hardware          CPU, RAM (if psutil is installed), NVIDIA GPU (via nvidia-smi)
  2. CT study size     average size of one study in a folder of your data
  3. Network           round-trip time (TCP connect) and upload speed
  4. Inference time    how long YOUR model/pipeline takes per CT study

At the end it prints a ready-to-run command for edge_experiments.py and saves
benchmark_results.json.

Usage (run it on the machine that will act as the edge node, at the clinic if possible)
    python measure_inputs.py --study-dir "D:\\data\\TCGA-OV" --model my_pipeline.py:run_study
    python measure_inputs.py --skip-network --study-dir "D:\\data\\TCGA-OV"
    python measure_inputs.py --rtt-host my.cloud.host.edu --upload-url https://my.cloud.host.edu/upload

Your model hook (--model path/to/file.py:function)
    The function takes ONE argument, the path of a study folder (str), and must run the
    WHOLE per-study pipeline you would run at the edge: DICOM load, segmentation, radiomics,
    FIGO classification. Load model weights once at the top of your file, not inside the
    function, so you time inference and not weight loading. Example my_pipeline.py:

        import my_segmentation_code as seg
        MODEL = seg.load_weights("weights.pt")          # loaded once at import
        def run_study(study_path: str):
            vol = seg.load_dicom(study_path)
            mask = MODEL.segment(vol)
            return seg.classify_figo(vol, mask)

Notes
  * Each immediate sub-folder of --study-dir is treated as one study. If there are none,
    the folder itself is one study.
  * The first call is timed separately (model load / GPU warm-up) and excluded from the mean.
  * Upload speed is measured against a public test endpoint (default Cloudflare). If it is
    blocked, measure with a speed test from the clinic network and pass the value by hand.
  * Round-trip time is the TCP connect time, which is about one network round trip.

Requires: Python 3.9+ standard library only (psutil optional, for RAM).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


# --------------------------------------------------------------------------
# 1. Hardware
# --------------------------------------------------------------------------
def hardware() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_threads": os.cpu_count(), "ram_gb": None}
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10)
        info["gpu"] = out.stdout.strip() or "none detected"
    except Exception:
        info["gpu"] = "none detected (nvidia-smi not found)"
    return info


# --------------------------------------------------------------------------
# 2. Study size
# --------------------------------------------------------------------------
def folder_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def find_studies(root: Path) -> list[Path]:
    subs = sorted(d for d in root.iterdir() if d.is_dir())
    return subs or [root]


def study_sizes_from(sizes: list[float]) -> dict:
    return {"n_studies": len(sizes), "mean_mb": round(statistics.mean(sizes), 1),
            "median_mb": round(statistics.median(sizes), 1),
            "min_mb": round(min(sizes), 1), "max_mb": round(max(sizes), 1)}


def study_sizes(studies: list[Path]) -> dict:
    sizes = []
    for i, s in enumerate(studies, 1):
        sizes.append(folder_mb(s))
        print(f"  scanned {i}/{len(studies)} study folders", end="\r", flush=True)
    print()
    return {"n_studies": len(sizes), "mean_mb": round(statistics.mean(sizes), 1),
            "median_mb": round(statistics.median(sizes), 1),
            "min_mb": round(min(sizes), 1), "max_mb": round(max(sizes), 1)}


# --------------------------------------------------------------------------
# 3. Network
# --------------------------------------------------------------------------
def measure_rtt(host: str, port: int = 443, n: int = 15) -> dict | None:
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=5):
                pass
        except OSError:
            continue
        t.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.1)
    if not t:
        return None
    t.sort()
    return {"host": host, "samples": len(t), "median_ms": round(statistics.median(t), 1),
            "p95_ms": round(t[min(len(t) - 1, int(0.95 * len(t)))], 1)}


def measure_uplink(url: str, mb: float, repeats: int = 3) -> dict:
    data = os.urandom(int(mb * 1e6))
    speeds = []
    for _ in range(repeats):
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/octet-stream", "User-Agent": "edge-benchmark"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
        except Exception as e:  # blocked, offline, or endpoint refused
            return {"url": url, "error": str(e)[:200]}
        speeds.append(mb * 8 / (time.perf_counter() - t0))
    return {"url": url, "upload_mb": mb, "mbps_median": round(statistics.median(speeds), 1),
            "mbps_all": [round(s, 1) for s in speeds]}


# --------------------------------------------------------------------------
# 4. Inference time
# --------------------------------------------------------------------------
def load_func(spec: str):
    file, func = spec.rsplit(":", 1)
    mod_spec = importlib.util.spec_from_file_location("user_model", file)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, func)


def time_inference(func, studies: list[Path], runs: int, max_studies: int) -> dict:
    chosen = studies[:max_studies]
    first_call, ok = None, []
    for s in chosen:  # probe call: start-up cost, caches, and skips studies that fail
        t0 = time.perf_counter()
        try:
            func(str(s))
        except Exception as e:
            print(f"  skipped {s.name}: {str(e)[:120]}")
            continue
        if first_call is None:
            first_call = time.perf_counter() - t0
        ok.append(s)
    if not ok:
        raise SystemExit("no study could be processed by the model function")
    times, per_study = [], []
    for s in ok:
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            func(str(s))
            ts.append(time.perf_counter() - t0)
        times.extend(ts)
        per_study.append({"study": s.name, "size_mb": round(folder_mb(s), 1),
                          "mean_s": round(statistics.mean(ts), 2),
                          "min_s": round(min(ts), 2), "max_s": round(max(ts), 2)})
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {"studies_timed": len(ok), "runs_per_study": runs,
            "first_call_s": round(first_call, 2), "mean_s": round(mean, 2),
            "std_s": round(std, 2), "cv": round(std / mean, 3) if mean else None,
            "min_s": round(min(times), 2), "max_s": round(max(times), 2),
            "per_study": per_study}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", help="folder with your CT studies (one sub-folder per study)")
    ap.add_argument("--model", help="path/to/file.py:function that runs the pipeline on one study")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per study (default 3)")
    ap.add_argument("--max-studies", type=int, default=3, help="studies to time (default 3)")
    ap.add_argument("--prefix", default="",
                    help="only treat sub-folders whose name starts with this as studies (e.g. TCGA-)")
    ap.add_argument("--pick-sizes", help="comma-separated target sizes in MB, e.g. 30,112,500,1500: "
                    "scans all studies and times the study nearest each target")
    ap.add_argument("--exclude", default="TCGA-61-2097,TCGA-10-0935",
                    help="study folder names never picked by --pick-sizes")
    ap.add_argument("--only", help="comma-separated study folder names to time (e.g. TCGA-04-1331,TCGA-13-0762); "
                    "default is the first --max-studies folders")
    ap.add_argument("--size-sample", type=int, default=0,
                    help="measure size on only N evenly spaced studies (faster on big folders)")
    ap.add_argument("--skip-network", action="store_true")
    ap.add_argument("--rtt-host", default="speed.cloudflare.com",
                    help="host to measure round-trip time to (use your real cloud host if known)")
    ap.add_argument("--upload-url", default="https://speed.cloudflare.com/__up")
    ap.add_argument("--upload-mb", type=float, default=20.0)
    ap.add_argument("--out", default="benchmark_results.json")
    a = ap.parse_args()

    res: dict = {"hardware": hardware()}
    print("Hardware:", json.dumps(res["hardware"], indent=2))

    studies: list[Path] = []
    pick_studies: list[Path] = []
    if a.study_dir:
        studies = find_studies(Path(a.study_dir))
        if a.prefix:
            studies = [s for s in studies if s.name.startswith(a.prefix)] or studies
        sized = studies
        if a.pick_sizes:
            sizes_by = {}
            for i, s in enumerate(studies, 1):
                sizes_by[s] = folder_mb(s)
                print(f"  scanned {i}/{len(studies)} study folders", end="\r", flush=True)
            print()
            bad = {x.strip() for x in a.exclude.split(",") if x.strip()}
            picks, pool = [], {s: m for s, m in sizes_by.items() if s.name not in bad}
            for target in [float(x) for x in a.pick_sizes.split(",")]:
                if not pool:
                    break
                best = min(pool, key=lambda s: abs(pool[s] - target))
                picks.append(best)
                pool.pop(best)
            pick_studies = picks
            a.only = None
            res["study_size"] = study_sizes_from(list(sizes_by.values()))
            res["picked"] = [{"study": s.name, "size_mb": round(sizes_by[s], 1)} for s in picks]
            print("Picked for timing:", ", ".join(f"{s.name} ({sizes_by[s]:.0f} MB)" for s in picks))
            sized = None
        elif a.size_sample and a.size_sample < len(studies):
            step = len(studies) / a.size_sample
            sized = [studies[int(i * step)] for i in range(a.size_sample)]
        if sized is not None:
            res["study_size"] = study_sizes(sized)
        print("\nStudy size:", json.dumps(res["study_size"]))
    else:
        print("\nStudy size: skipped (give --study-dir)")

    if not a.skip_network:
        print("\nMeasuring network (about 30 s) ...")
        res["rtt"] = measure_rtt(a.rtt_host)
        print("RTT:", res["rtt"] or "could not connect")
        res["uplink"] = measure_uplink(a.upload_url, a.upload_mb)
        print("Uplink:", res["uplink"])

    if a.model and studies:
        print("\nTiming your pipeline ...")
        timing_studies = pick_studies if a.pick_sizes else studies
        if a.pick_sizes:
            a.max_studies = max(a.max_studies, len(timing_studies))
        if a.only:
            wanted = [x.strip() for x in a.only.split(",") if x.strip()]
            timing_studies = [s for s in studies if s.name in wanted]
            missing = set(wanted) - {s.name for s in timing_studies}
            if missing:
                print("WARNING: not found in --study-dir:", ", ".join(sorted(missing)))
            if not timing_studies:
                raise SystemExit("none of the --only studies were found")
        res["inference"] = time_inference(load_func(a.model), timing_studies, a.runs, a.max_studies)
        inf = res["inference"]
        print("\nPer-study timing (steady state, first call excluded):")
        print(f"  {'study':<18}{'size MB':>9}{'mean s':>9}{'min s':>8}{'max s':>8}")
        for r in inf["per_study"]:
            print(f"  {r['study']:<18}{r['size_mb']:>9}{r['mean_s']:>9}{r['min_s']:>8}{r['max_s']:>8}")
        print(f"First call (start-up, model set-up): {inf['first_call_s']} s")
        print(f"Overall mean {inf['mean_s']} s, std {inf['std_s']} s, cv {inf['cv']}"
              "   (mixes study sizes: use the row nearest your typical study for the paper)")
    else:
        print("\nInference timing skipped (give both --model and --study-dir)")

    # ready-to-run command
    flags = []
    if "study_size" in res:
        flags.append(f"--study-mb {res['study_size']['mean_mb']:.0f}")
    if res.get("uplink", {}).get("mbps_median"):
        flags.append(f"--wan-mbps {res['uplink']['mbps_median']:.0f}")
    if res.get("rtt"):
        flags.append(f"--rtt-ms {res['rtt']['median_ms']:.0f}")
    if "inference" in res:
        flags.append(f"--edge-infer-s {res['inference']['mean_s']:.1f}")
    print("\n" + "=" * 70)
    if flags:
        print("Next, run the experiments with your measured values:\n")
        print("  python edge_experiments.py " + " ".join(flags) + " --out results_measured")
    print("\nStill assumed (not measured here): cloud inference time (--cloud-infer-s),"
          "\noutage probability (--outage-prob; log connectivity for a week or ask IT),"
          "\nand the inference-time spread (edit cv in class P if your measured cv differs).")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
