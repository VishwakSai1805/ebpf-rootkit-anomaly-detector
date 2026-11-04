"""Train the detector on synthetic benign traffic: ``python -m ml.train --out build/model.npz``."""
from __future__ import annotations

import argparse
import time

from .dataset import benign_training_matrix
from .detector import AnomalyDetector
from .pipeline import DEFAULT_MODEL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    ap.add_argument("--sessions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--percentile", type=float, default=99.7)
    a = ap.parse_args(argv)
    t0 = time.time()
    X = benign_training_matrix(a.sessions, seed=1234 + a.seed)
    m = AnomalyDetector(seed=a.seed, percentile=a.percentile).fit(X)
    m.save(a.out)
    print(f"trained on {len(X)} benign windows in {time.time() - t0:.1f}s -> {a.out} "
          f"(thr_if={m.thr_if:.4f}, thr_ae={m.thr_ae:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
