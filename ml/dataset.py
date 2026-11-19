"""Turns labelled sessions into window feature matrices using the production featurizer."""
from __future__ import annotations

import numpy as np

from collector.scenarios import Session, generate_sessions
from .features import FEATURE_DIM, WindowFeaturizer

BENIGN = "benign"


def session_windows(session: Session, window: int = 32, stride: int = 1):
    """Yield (features, label, pid, ts_ns) for every ``stride``-th scored window of a session."""
    wf = WindowFeaturizer(window=window)
    proc = frozenset(session.pids - session.hidden_pids)
    k = 0
    for ev in session.events:
        r = wf.push(ev, proc)
        if r is None:
            continue
        k += 1
        if k % stride:
            continue
        x, w = r
        bad = next((e.label for e in w if e.label and e.label != BENIGN), BENIGN)
        yield x, bad, ev.pid, ev.ts_ns


def build_matrix(sessions: list[Session], window: int = 32, stride: int = 1):
    X, y = [], []
    for s in sessions:
        for x, lab, _, _ in session_windows(s, window, stride):
            X.append(x)
            y.append(lab)
    if not X:
        return np.zeros((0, FEATURE_DIM), np.float32), np.array([], dtype=object)
    return np.stack(X), np.array(y, dtype=object)


def benign_training_matrix(n_sessions: int = 500, seed: int = 1234, stride: int = 2):
    X, _ = build_matrix(generate_sessions(n_sessions, 0, seed), stride=stride)
    return X
