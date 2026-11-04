"""Streaming detection engine: events in, alerts out."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from collector.events import Event
from .detector import AnomalyDetector
from .features import WindowFeaturizer

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "build" / "model.npz"


@dataclass
class Alert:
    ts_ns: int
    pid: int
    comm: str
    uid: int
    score: float
    if_score: float
    ae_error: float
    categories: list[str]
    recent: list[str] = field(default_factory=list)
    truth: str = ""  # ground-truth label when known (mock/replay); informational only

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @property
    def severity(self) -> str:
        return "critical" if self.score >= 2.0 else "high" if self.score >= 1.3 else "medium"


class StreamingDetector:
    """Feeds events through per-PID windows, scores each window and emits de-duplicated alerts."""

    def __init__(self, model: AnomalyDetector, window: int = 32, cooldown_ns: int = 5_000_000_000):
        self.model = model
        self.feat = WindowFeaturizer(window=window)
        self.cooldown_ns = cooldown_ns
        self._last_alert: dict[tuple[int, str], int] = {}
        self.windows_scored = 0
        self.latencies_ns: list[int] = []
        self.keep_latencies = False

    def process(self, ev: Event, proc: frozenset[int] | None = None) -> Alert | None:
        r = self.feat.push(ev, proc)
        if r is None:
            return None
        x, w = r
        t0 = time.perf_counter_ns()
        det = self.model.detect(x)
        if self.keep_latencies:
            self.latencies_ns.append(time.perf_counter_ns() - t0)
        self.windows_scored += 1
        if not det.anomaly:
            return None
        key = (ev.pid, det.categories[0])
        last = self._last_alert.get(key)
        if last is not None and ev.ts_ns - last < self.cooldown_ns:
            return None
        self._last_alert[key] = ev.ts_ns
        if len(self._last_alert) > 4096:
            self._last_alert.clear()
        truth = next((e.label for e in reversed(w) if e.label and e.label != "benign"), "")
        return Alert(ev.ts_ns, ev.pid, ev.comm, ev.uid, round(det.score, 3), round(det.if_score, 4),
                     round(det.ae_error, 4), det.categories,
                     [f"{e.type}:{e.data or e.arg0}" for e in list(w)[-6:]], truth)


def load_or_train(path: str | Path | None = None, retrain: bool = False, seed: int = 0,
                  n_sessions: int = 500) -> AnomalyDetector:
    from .dataset import benign_training_matrix
    path = Path(path) if path else DEFAULT_MODEL
    if path.is_file() and not retrain:
        try:
            return AnomalyDetector.load(path)
        except (ValueError, KeyError):
            pass
    model = AnomalyDetector(seed=seed).fit(benign_training_matrix(n_sessions, seed=1234 + seed))
    try:
        model.save(path)
    except OSError:
        pass
    return model
