#!/usr/bin/env python3
"""
Simulation of the software-defined edge IIoT pipeline for one CT-based ovarian
cancer screening visit, repeated for many visits.

Per visit it runs, from "scan complete" to "result ready for the clinician":
    policy_check (control plane) -> lan_transfer -> deidentify -> inference
and compares two deployments:
    EDGE  : inference on the clinic edge node, only a small result leaves the site
    CLOUD : full study uploaded over the WAN, inference in the cloud

Honest scope
  * "measured" stages (policy check, de-identification, report, audit hashing)
    execute real code and are timed with perf_counter.
  * "modeled" stages (network transfer, GPU inference time, clinician review)
    are drawn from simple parameterised distributions; set them from your own
    benchmarks before quoting numbers in a paper.
  * de-identification here is a simplified tag-level demo, NOT a certified
    DICOM profile. In production use pydicom with a standard profile.
  * The model is a stub. Plug in your own with --model my_model.py:predict
    where predict(pseudonym: str) -> list of 4 probabilities (FIGO I..IV).
  * The clinician always confirms; the AI output is decision support only.

Usage
    python visit_pipeline_sim.py
    python visit_pipeline_sim.py --visits 500 --wan-mbps 50 --outage-prob 0.1
    python visit_pipeline_sim.py --model my_segmentation.py:predict

Outputs (in --out, default ./sim_results): stage_latency.csv, audit.jsonl, summary.json
Requires only the Python standard library (3.9+).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import importlib.util
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

FIGO = ["I", "II", "III", "IV"]


@dataclass
class Config:
    visits: int = 200
    seed: int = 1
    study_mb: float = 200.0        # size of one CT study
    lan_mbps: float = 1000.0       # scanner -> edge node
    wan_mbps: float = 100.0        # clinic -> cloud uplink
    rtt_ms: float = 40.0           # clinic <-> cloud round trip
    outage_prob: float = 0.05      # chance the WAN is down for a visit
    outage_wait_s: tuple = (60.0, 600.0)  # how long an outage lasts (min, max)
    edge_infer_s: float = 12.0     # modeled inference time on edge hardware
    cloud_infer_s: float = 4.0     # modeled inference time on cloud GPU
    review_routine_s: float = 90.0
    review_flagged_s: float = 240.0
    entropy_threshold: float = 0.6  # normalised entropy above this -> flagged
    min_confidence: float = 0.6
    model_mb: float = 100.0        # size of one federated weight update
    visits_per_round: int = 50     # visits between federated updates
    secret: bytes = b"change-me-in-production"


# --------------------------------------------------------------------------
# Tamper-evident audit log (hash chain)
# --------------------------------------------------------------------------
class AuditLog:
    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._prev = "0" * 64

    def append(self, visit: str, stage: str, detail: str) -> None:
        body = {"seq": len(self.entries), "visit": visit, "stage": stage,
                "detail": detail, "prev": self._prev}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.entries.append({**body, "hash": digest})
        self._prev = digest

    def verify(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            body = {k: e[k] for k in ("seq", "visit", "stage", "detail", "prev")}
            ok = e["prev"] == prev and e["hash"] == hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()).hexdigest()
            if not ok:
                return False
            prev = e["hash"]
        return True

    def save(self, path: Path) -> None:
        path.write_text("\n".join(json.dumps(e) for e in self.entries) + "\n")


# --------------------------------------------------------------------------
# Control plane: signed token + QoS slice
# --------------------------------------------------------------------------
def issue_token(secret: bytes, visit: str, scope: str, expires: float) -> tuple[str, str]:
    payload = f"{visit}|{scope}|{expires}"
    return payload, hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()


def policy_check(cfg: Config, visit: str, consent: bool) -> str:
    now = time.time()
    payload, sig = issue_token(cfg.secret, visit, "ct:upload", now + 300)
    expected = hmac.new(cfg.secret, payload.encode(), hashlib.sha256).hexdigest()
    _, scope, exp = payload.split("|")
    if not (consent and hmac.compare_digest(sig, expected)
            and scope == "ct:upload" and float(exp) > now):
        raise PermissionError("policy check failed")
    return "clinical-imaging"  # QoS slice assigned by the SDN controller


# --------------------------------------------------------------------------
# Edge: de-identification (simplified demo)
# --------------------------------------------------------------------------
def deidentify(header: dict, secret: bytes) -> tuple[dict, str]:
    pseudonym = hmac.new(secret, header["PatientID"].encode(),
                         hashlib.sha256).hexdigest()[:16]
    clean = {k: v for k, v in header.items()
             if k not in {"PatientName", "PatientBirthDate", "InstitutionName",
                          "ReferringPhysicianName"}}
    clean["PatientID"] = pseudonym
    clean["StudyDate"] = clean["StudyDate"][:4] + "0101"  # keep year only
    return clean, pseudonym


# --------------------------------------------------------------------------
# Inference: stub model or user plug-in
# --------------------------------------------------------------------------
def stub_predict(pseudonym: str) -> list[float]:
    r = random.Random(pseudonym)
    logits = [r.gauss(0, 2.2) for _ in FIGO]  # sharper = more confident stub
    z = [math.exp(v) for v in logits]
    s = sum(z)
    return [v / s for v in z]


def load_model(spec: str | None) -> Callable[[str], list[float]]:
    if not spec:
        return stub_predict
    file, func = spec.rsplit(":", 1)
    mod_spec = importlib.util.spec_from_file_location("user_model", file)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, func)


def norm_entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p))


# --------------------------------------------------------------------------
# Modeled latency helpers
# --------------------------------------------------------------------------
def transfer_ms(mb: float, mbps: float, rtt_ms: float, rng: random.Random) -> float:
    return (mb * 8 / mbps * 1000 + rtt_ms) * rng.uniform(0.9, 1.4)


def jitter_ms(rng: random.Random, mean_s: float, cv: float = 0.15) -> float:
    return max(0.01, rng.gauss(mean_s, mean_s * cv)) * 1000


def timed(fn: Callable, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return out, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------
# One visit
# --------------------------------------------------------------------------
def simulate_visit(i: int, cfg: Config, rng: random.Random, predict: Callable,
                   audit: AuditLog, rows: list[dict]) -> dict:
    vid = f"V{i:04d}"

    def rec(path: str, stage: str, kind: str, ms: float) -> None:
        rows.append({"visit": vid, "path": path, "stage": stage,
                     "kind": kind, "ms": round(ms, 3)})

    header = {"PatientName": f"Patient^{i}", "PatientID": f"MRN{100000 + i}",
              "PatientBirthDate": "19700101", "InstitutionName": "Clinic A",
              "ReferringPhysicianName": "Dr X", "StudyDate": "20261015",
              "Modality": "CT"}

    # control plane (shared by both paths)
    _, t_pol = timed(policy_check, cfg, vid, True)
    audit.append(vid, "policy_check", "slice=clinical-imaging")
    lan = transfer_ms(cfg.study_mb, cfg.lan_mbps, 1.0, rng)
    (clean, pseud), t_deid = timed(deidentify, header, cfg.secret)
    audit.append(vid, "deidentify", f"pseudonym={pseud}")
    for p in ("edge", "cloud"):
        rec(p, "policy_check", "measured", t_pol)
        rec(p, "lan_transfer", "modeled", lan)
        rec(p, "deidentify", "measured", t_deid)

    # edge path: inference next to the scanner
    probs, _ = timed(predict, pseud)
    inf_edge = jitter_ms(rng, cfg.edge_infer_s)
    rec("edge", "inference", "modeled", inf_edge)
    edge_total = t_pol + lan + t_deid + inf_edge

    # cloud path: WAN upload + cloud inference + result back, with outages
    wan = transfer_ms(cfg.study_mb, cfg.wan_mbps, cfg.rtt_ms, rng)
    outage = rng.random() < cfg.outage_prob
    wait = rng.uniform(*cfg.outage_wait_s) * 1000 if outage else 0.0
    inf_cloud = jitter_ms(rng, cfg.cloud_infer_s)
    ret = cfg.rtt_ms * rng.uniform(0.9, 1.4)
    rec("cloud", "wan_upload", "modeled", wan + wait)
    rec("cloud", "inference", "modeled", inf_cloud)
    rec("cloud", "result_return", "modeled", ret)
    cloud_total = t_pol + lan + t_deid + wan + wait + inf_cloud + ret

    # clinician review (same for both; AI is decision support only)
    top = max(range(4), key=lambda k: probs[k])
    flagged = probs[top] < cfg.min_confidence or norm_entropy(probs) > cfg.entropy_threshold
    review = jitter_ms(rng, cfg.review_flagged_s if flagged else cfg.review_routine_s)
    corrected = rng.random() > (0.70 if flagged else 0.95)
    audit.append(vid, "clinician_review",
                 f"stage={FIGO[top]} flagged={flagged} corrected={corrected}")

    (_, t_rep) = timed(lambda: json.dumps({"visit": vid, "figo": FIGO[top],
                                           "probs": [round(p, 3) for p in probs]}))
    rec("edge", "report", "measured", t_rep)
    audit.append(vid, "report", "released to patient portal")

    return {"edge_ms": edge_total, "cloud_ms": cloud_total, "outage": outage,
            "flagged": flagged, "corrected": corrected, "review_ms": review}


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarise(cfg: Config, results: list[dict], audit_ok: bool) -> dict:
    edge = [r["edge_ms"] / 1000 for r in results]
    cloud = [r["cloud_ms"] / 1000 for r in results]
    out_cloud = [r["cloud_ms"] / 1000 for r in results if r["outage"]]
    n = len(results)
    cloud_mb = n * cfg.study_mb
    edge_mb = n * 0.002 + (n / cfg.visits_per_round) * cfg.model_mb
    return {
        "visits": n,
        "time_to_result_s": {
            "edge_p50": round(statistics.median(edge), 1), "edge_p95": round(pct(edge, .95), 1),
            "cloud_p50": round(statistics.median(cloud), 1), "cloud_p95": round(pct(cloud, .95), 1),
            "cloud_p50_during_outage": round(statistics.median(out_cloud), 1) if out_cloud else None,
        },
        "outage_visits": sum(r["outage"] for r in results),
        "flagged_for_careful_review_pct": round(100 * sum(r["flagged"] for r in results) / n, 1),
        "clinician_corrections_pct": round(100 * sum(r["corrected"] for r in results) / n, 1),
        "uplink_gb": {"edge": round(edge_mb / 1000, 2), "cloud": round(cloud_mb / 1000, 2),
                      "saved_pct": round(100 * (1 - edge_mb / cloud_mb), 1)},
        "audit_chain_verified": audit_ok,
        "config": {k: (list(v) if isinstance(v, tuple) else (v.decode() if isinstance(v, bytes) else v))
                   for k, v in asdict(cfg).items() if k != "secret"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sim_results")
    ap.add_argument("--model", default=None, help="path/to/file.py:function")
    ap.add_argument("--visits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--study-mb", type=float, default=200.0)
    ap.add_argument("--wan-mbps", type=float, default=100.0)
    ap.add_argument("--rtt-ms", type=float, default=40.0)
    ap.add_argument("--outage-prob", type=float, default=0.05)
    ap.add_argument("--edge-infer-s", type=float, default=12.0)
    ap.add_argument("--cloud-infer-s", type=float, default=4.0)
    a = ap.parse_args()

    cfg = Config(visits=a.visits, seed=a.seed, study_mb=a.study_mb, wan_mbps=a.wan_mbps,
                 rtt_ms=a.rtt_ms, outage_prob=a.outage_prob,
                 edge_infer_s=a.edge_infer_s, cloud_infer_s=a.cloud_infer_s)
    rng = random.Random(cfg.seed)
    predict = load_model(a.model)
    audit, rows = AuditLog(), []
    results = [simulate_visit(i, cfg, rng, predict, audit, rows) for i in range(cfg.visits)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stage_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["visit", "path", "stage", "kind", "ms"])
        w.writeheader()
        w.writerows(rows)
    audit.save(out / "audit.jsonl")
    summary = summarise(cfg, results, audit.verify())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    t = summary["time_to_result_s"]
    print(f"visits: {summary['visits']}   outage visits: {summary['outage_visits']}")
    print(f"time to result (s)   edge p50 {t['edge_p50']} / p95 {t['edge_p95']}"
          f"   cloud p50 {t['cloud_p50']} / p95 {t['cloud_p95']}")
    print(f"uplink (GB)          edge {summary['uplink_gb']['edge']}"
          f"   cloud {summary['uplink_gb']['cloud']}   saved {summary['uplink_gb']['saved_pct']}%")
    print(f"flagged for careful review: {summary['flagged_for_careful_review_pct']}%"
          f"   audit chain verified: {summary['audit_chain_verified']}")
    print(f"files written to {out}/")


if __name__ == "__main__":
    main()
