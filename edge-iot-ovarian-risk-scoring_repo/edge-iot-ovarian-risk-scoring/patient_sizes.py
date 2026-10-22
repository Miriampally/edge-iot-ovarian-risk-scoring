#!/usr/bin/env python3
"""
List the size of every patient folder in the TCGA-OV DICOM folder, save a CSV, and show
the patients whose size is closest to a target (default 112 MB, the median in the paper).

Why: (1) pick a patient near the median size to time with my_pipeline.py;
     (2) check the folder count (your notes say 143 patient folders, the benchmark counted 144);
     any folder whose name does not start with "TCGA-" is flagged.

Works in Command Prompt or PowerShell (standard library only).

Usage
    python patient_sizes.py
    python patient_sizes.py --target 534
    python patient_sizes.py "D:\\other\\path\\TCGA-OV" --out sizes.csv

Size is the WHOLE patient folder (all series), not only the series the pipeline selects.
"""
import argparse
import csv
import os
import statistics
from pathlib import Path

DEFAULT = os.environ.get("TCGA_OV_DICOM_ROOT", "TCGA-OV")  # folder with one sub-folder per patient


def folder_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=DEFAULT)
    ap.add_argument("--target", type=float, default=112.0, help="target size in MB")
    ap.add_argument("--top", type=int, default=6, help="how many closest patients to show")
    ap.add_argument("--out", default="patient_sizes.csv")
    a = ap.parse_args()

    dirs = sorted(d for d in Path(a.root).iterdir() if d.is_dir())
    rows = []
    for i, d in enumerate(dirs, 1):
        rows.append({"patient": d.name, "mb": round(folder_mb(d), 1),
                     "patient_style_name": d.name.startswith("TCGA-")})
        print(f"  scanned {i}/{len(dirs)}", end="\r", flush=True)
    print()
    rows.sort(key=lambda r: r["mb"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "mb", "patient_style_name"])
        w.writeheader()
        w.writerows(rows)

    pat = [r for r in rows if r["patient_style_name"]]
    for label, rs in (("all folders      ", rows), ("PATIENT folders  ", pat)):
        sz = [r["mb"] for r in rs]
        print(f"{label} n={len(rs)}   median {statistics.median(sz):.1f} MB   "
              f"mean {statistics.mean(sz):.1f} MB   min {min(sz)}   max {max(sz)}")
    odd = [r["patient"] + f" ({r['mb']} MB)" for r in rows if not r["patient_style_name"]]
    print("folders not named TCGA-* (excluded from the PATIENT line):", odd if odd else "none")
    print("Use the PATIENT line in the paper.")
    print(f"\nClosest to {a.target:g} MB (patient folders only):")
    for r in sorted(pat, key=lambda r: abs(r["mb"] - a.target))[: a.top]:
        print(f"  {r['patient']:<16}{r['mb']:>8.1f} MB")
    print(f"\nSaved {a.out}")
    print("Avoid TCGA-61-2097 and TCGA-10-0935 (excluded in config.py) when picking.")


if __name__ == "__main__":
    main()
