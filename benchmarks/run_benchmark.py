"""Benchmark: precision/recall/F1, per-class recall, time-to-detect and latency.

    python -m benchmarks.run_benchmark            # writes reports/benchmark_results.{md,json}

Train corpus: benign only (seed 1234). Test corpora: 3 independent seeds, each 400 benign sessions +
100 sessions per attack class. A session counts as *detected* only if a window that contains an
attack-labelled event is flagged (alerts in the benign prefix of an attack session do not count).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn

from collector.scenarios import ATTACK_KINDS, generate_sessions
from collector.sources import MockSource
from ml.dataset import benign_training_matrix, session_windows
from ml.detector import AnomalyDetector, categorize
from ml.features import WindowFeaturizer, extract
from ml.pipeline import StreamingDetector

REPORTS = Path(__file__).resolve().parent.parent / "reports"
EXPECTED_CAT = {"privesc": "privilege_escalation", "reverse_shell": "reverse_shell", "ld_preload": "ld_preload_injection",
                "ptrace_injection": "ptrace_injection", "hidden_pid": "hidden_process"}
VARIANTS = ("fused", "isolation_forest", "autoencoder")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(model: AnomalyDetector, sessions):
    """Returns per-variant window- and session-level confusion counts plus per-class details."""
    res = {v: {"win": dict(tp=0, fp=0, fn=0, tn=0), "sess": dict(tp=0, fp=0, fn=0, tn=0),
               "class_hit": {k: 0 for k in ATTACK_KINDS}, "class_n": {k: 0 for k in ATTACK_KINDS}} for v in VARIANTS}
    ttd = {k: [] for k in ATTACK_KINDS}
    cat_ok = {k: [0, 0] for k in ATTACK_KINDS}
    for s in sessions:
        rows = list(session_windows(s))
        if not rows:
            continue
        X = np.stack([r[0] for r in rows])
        atk = np.array([r[1] != "benign" for r in rows])
        fused, s_if, s_ae = model.score_batch(X)
        flags = {"fused": fused >= 1, "isolation_forest": s_if > model.thr_if, "autoencoder": s_ae > model.thr_ae}
        for v, fl in flags.items():
            w = res[v]["win"]
            w["tp"] += int((fl & atk).sum()); w["fp"] += int((fl & ~atk).sum())
            w["fn"] += int((~fl & atk).sum()); w["tn"] += int((~fl & ~atk).sum())
            c = res[v]["sess"]
            if s.is_attack:
                hit = bool((fl & atk).any())
                res[v]["class_n"][s.label] += 1
                res[v]["class_hit"][s.label] += hit
                c["tp" if hit else "fn"] += 1
            else:
                c["fp" if fl.any() else "tn"] += 1
        if s.is_attack:
            fl = flags["fused"]
            first_atk = int(np.argmax(atk))
            idx = np.nonzero(fl & atk)[0]
            if len(idx):
                ttd[s.label].append(int(idx[0]) - first_atk)
                cats = set()
                for i in idx:
                    cats.update(categorize(X[i]))
                cat_ok[s.label][1] += 1
                cat_ok[s.label][0] += EXPECTED_CAT[s.label] in cats
    return res, ttd, cat_ok


def latency_profile(model: AnomalyDetector, n_windows: int = 4000):
    src = MockSource(seed=99, attack_rate=0.15)
    wf = WindowFeaturizer()
    wins, vecs = [], []
    for ev in src:
        r = wf.push(ev, src.proc_snapshot())
        if r:
            wins.append(list(r[1])); vecs.append(r[0])
            if len(wins) >= n_windows:
                break
    for x in vecs[:50]:
        model.detect(x)

    def timeit(fn, items):
        out = []
        for it in items:
            t = time.perf_counter_ns(); fn(it); out.append((time.perf_counter_ns() - t) / 1e3)
        a = np.array(out)
        return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max())}
    prof = {"feature_extraction": timeit(lambda w: extract(w, 0), wins),
            "isolation_forest": timeit(model.if_scores, vecs),
            "autoencoder": timeit(model.ae_errors, vecs),
            "inference_total (IF+AE+categories)": timeit(model.detect, vecs)}
    # full streaming pipeline throughput
    src2 = MockSource(seed=7, attack_rate=0.08, max_events=40000)
    sd = StreamingDetector(model)
    evs = list(src2)
    t = time.perf_counter()
    for ev in evs:
        sd.process(ev, src2.proc_snapshot())
    el = time.perf_counter() - t
    prof["_throughput_events_per_s"] = len(evs) / el
    # end-to-end per-event (featurise + score) distribution
    sd2 = StreamingDetector(model)
    lat = []
    for ev in evs[:8000]:
        t0 = time.perf_counter_ns()
        r = sd2.feat.push(ev, src2.proc_snapshot())
        if r:
            model.detect(r[0])
            lat.append((time.perf_counter_ns() - t0) / 1e3)
    a = np.array(lat)
    prof["end_to_end_per_window"] = {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                                     "p99": float(np.percentile(a, 99)), "max": float(a.max())}
    return prof


def fmt(x):
    return f"{x:.3f}"


def run(seeds=(777, 778, 779), n_benign=400, per_kind=100, train_sessions=500, out_dir=REPORTS) -> dict:
    t0 = time.time()
    model = AnomalyDetector(seed=0).fit(benign_training_matrix(train_sessions, seed=1234))
    agg = None
    ttd_all = {k: [] for k in ATTACK_KINDS}
    cat_all = {k: [0, 0] for k in ATTACK_KINDS}
    per_seed = []
    for sd in seeds:
        res, ttd, cat = evaluate(model, generate_sessions(n_benign, per_kind, seed=sd))
        per_seed.append({v: prf(res[v]["sess"]["tp"], res[v]["sess"]["fp"], res[v]["sess"]["fn"])[2] for v in VARIANTS})
        for k in ATTACK_KINDS:
            ttd_all[k] += ttd[k]
            cat_all[k][0] += cat[k][0]; cat_all[k][1] += cat[k][1]
        if agg is None:
            agg = res
        else:
            for v in VARIANTS:
                for lvl in ("win", "sess"):
                    for key in agg[v][lvl]:
                        agg[v][lvl][key] += res[v][lvl][key]
                for k in ATTACK_KINDS:
                    agg[v]["class_hit"][k] += res[v]["class_hit"][k]
                    agg[v]["class_n"][k] += res[v]["class_n"][k]
    lat = latency_profile(model)
    out = {"agg": agg, "ttd": {k: v for k, v in ttd_all.items()}, "cat": cat_all, "per_seed_f1": per_seed,
           "latency": lat, "config": {"seeds": list(seeds), "benign": n_benign, "per_kind": per_kind,
                                      "train_sessions": train_sessions, "thr_if": model.thr_if, "thr_ae": model.thr_ae,
                                      "percentile": model.percentile},
           "runtime_s": time.time() - t0}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_results.json").write_text(json.dumps(out, indent=1, default=float))
    (out_dir / "benchmark_results.md").write_text(to_markdown(out))
    return out


def to_markdown(o: dict) -> str:
    agg, cfg, lat = o["agg"], o["config"], o["latency"]
    L = ["# Benchmark results", "",
         "Generated by `python -m benchmarks.run_benchmark` (deterministic; seeds fixed).", "",
         "> **Read this first.** Traffic is *synthetic* (see `collector/scenarios.py`): benign workloads include hard negatives "
         "(sudo/su, gdb/strace, `LD_PRELOAD` dev tools, sshd privilege drops, shell pipelines) but the attack chains are "
         "idealised. These numbers measure separability on this generator, not production false-positive rates on a real fleet.", "",
         "## Setup", "",
         f"- Training: {cfg['train_sessions']} benign sessions only (unsupervised); thresholds = p{cfg['percentile']} of held-out benign scores "
         f"(IF {cfg['thr_if']:.4f}, AE {cfg['thr_ae']:.4f}).",
         f"- Test: {len(cfg['seeds'])} independent corpora (seeds {cfg['seeds']}), each {cfg['benign']} benign + {cfg['per_kind']} sessions "
         f"x {len(ATTACK_KINDS)} attack classes. Pooled: {cfg['benign'] * len(cfg['seeds'])} benign / "
         f"{cfg['per_kind'] * len(cfg['seeds']) * len(ATTACK_KINDS)} attack sessions.",
         "- *Session detected* = a window containing an attack event is flagged. *Session false positive* = any window of a benign session flagged "
         "(a strict criterion: sessions have dozens of windows).",
         f"- Environment: Python {platform.python_version()}, numpy {np.__version__}, scikit-learn {sklearn.__version__}, "
         f"{platform.machine()}; total benchmark runtime {o['runtime_s']:.0f}s.", ""]
    L += ["## Session-level detection (attack vs. benign)", "",
          "| detector | precision | recall | F1 | benign-session FPR | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        c = agg[v]["sess"]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        L.append(f"| {v.replace('_', ' ')} | {fmt(p)} | {fmt(r)} | {fmt(f)} | {fmt(c['fp'] / max(c['fp'] + c['tn'], 1))} | "
                 f"{c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
    f1s = [x["fused"] for x in o["per_seed_f1"]]
    L += ["", f"Fused F1 per test seed: {', '.join(fmt(x) for x in f1s)} (spread {max(f1s) - min(f1s):.3f}).", "",
          "## Per-class recall (session level)", "",
          "| attack class | " + " | ".join(v.replace("_", " ") for v in VARIANTS) + " | median events-to-detect | attribution correct |",
          "|---|" + "---|" * (len(VARIANTS) + 2)]
    for k in ATTACK_KINDS:
        n = agg["fused"]["class_n"][k]
        rec = " | ".join(fmt(agg[v]["class_hit"][k] / n) for v in VARIANTS)
        t = o["ttd"][k]
        ca = o["cat"][k]
        L.append(f"| {k} (n={n}) | {rec} | {np.median(t):.0f} | {ca[0] / max(ca[1], 1):.2f} |" if t else f"| {k} | {rec} | - | - |")
    L += ["", "*events-to-detect*: scored windows between the first attack event and the first alert (0 = alert on the first attack window). "
          "*attribution*: the expected category appears among the explanations of the flagged windows.", "",
          "## Window-level detection", "",
          "A window is positive if it contains any attack-labelled event.", "",
          "| detector | precision | recall | F1 | window FPR |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        c = agg[v]["win"]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        L.append(f"| {v.replace('_', ' ')} | {fmt(p)} | {fmt(r)} | {fmt(f)} | {c['fp'] / max(c['fp'] + c['tn'], 1):.4f} |")
    L += ["", "Window recall is lower than session recall by design: early windows contain only the first (still ambiguous) attack step.", "",
          "## Inference latency (microseconds per window)", "",
          "| stage | p50 | p95 | p99 | max |", "|---|---|---|---|---|"]
    for k, v in lat.items():
        if isinstance(v, dict):
            L.append(f"| {k.replace('_', ' ')} | {v['p50']:.0f} | {v['p95']:.0f} | {v['p99']:.0f} | {v['max']:.0f} |")
    inf = lat["inference_total (IF+AE+categories)"]["p99"]
    e2e = lat["end_to_end_per_window"]["p99"]
    L += ["", f"Pipeline throughput (feature extraction + scoring, single thread): **{lat['_throughput_events_per_s']:.0f} events/s**.", "",
          f"Budget check: model inference p99 = {inf:.0f} us ({'PASS' if inf < 1000 else 'FAIL'} < 1000 us); "
          f"end-to-end per window p99 = {e2e:.0f} us ({'PASS' if e2e < 1000 else 'FAIL'} < 1000 us). Latency depends on the host CPU and load.", "",
          "## Observations / limitations", "",
          "- On this synthetic data the Isolation Forest alone has low recall (attacks live in a handful of sparse behavioural columns) and the "
          "autoencoder does nearly all the work; compare the `fused` and `autoencoder` rows: OR-fusing the forest adds its false positives "
          "while adding no session-level recall beyond the autoencoder. It is kept as an independent second opinion for ptrace-heavy windows "
          "and for real data whose structure may differ - tune or drop it for your environment.",
          "- Residual false positives come from the intentionally hard benign classes (gdb/strace sessions, `LD_PRELOAD` dev tooling). "
          "`ld_preload` recall is the lowest because one attack variant (a single `LD_PRELOAD` exec, no `ld.so.preload` write) is "
          "behaviourally identical to the benign dev-tool case.",
          "- Hidden-PID detection requires a trustworthy `/proc` snapshot and a grace period (1 s) to absorb fork/exit races.",
          "- Real kernels produce much higher event volume and noisier context than this generator; re-calibrate thresholds on your own benign baseline "
          "(`python -m ml.train`) before relying on alerts.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller corpora (for smoke tests)")
    ap.add_argument("--out", default=str(REPORTS))
    a = ap.parse_args(argv)
    kw = dict(seeds=(777,), n_benign=60, per_kind=15, train_sessions=200) if a.quick else {}
    o = run(out_dir=a.out, **kw)
    c = o["agg"]["fused"]["sess"]
    p, r, f = prf(c["tp"], c["fp"], c["fn"])
    print(f"fused session-level: precision={p:.3f} recall={r:.3f} F1={f:.3f}; wrote {a.out}/benchmark_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
