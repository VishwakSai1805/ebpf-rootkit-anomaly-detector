"""Event sources: Live (eBPF loader), Replay (JSONL) and Mock (synthetic), plus ``auto_source``."""
from __future__ import annotations

import heapq
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

from .events import Event
from .scenarios import ATTACK_KINDS, make_session, random_benign_kind

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OBJ = ROOT / "bpf" / "build" / "probes.bpf.o"
DEFAULT_LOADER = ROOT / "bpf" / "build" / "loader"


class SourceUnavailable(RuntimeError):
    pass


class EventSource:
    name = "base"

    def __iter__(self) -> Iterator[Event]:  # pragma: no cover - interface
        raise NotImplementedError

    def proc_snapshot(self) -> frozenset[int] | None:
        """PIDs visible in /proc (or its replayed/mocked equivalent); None = unknown."""
        return None

    def close(self) -> None:
        pass


def read_proc_pids() -> frozenset[int]:
    try:
        return frozenset(int(n) for n in os.listdir("/proc") if n.isdigit())
    except OSError:
        return frozenset()


class ReplaySource(EventSource):
    """Replays a JSONL capture. Lines with ``"type": "proc_snapshot"`` carry ``"pids"`` lists."""
    name = "replay"

    def __init__(self, path: str | os.PathLike, realtime: bool = False):
        self.path = Path(path)
        self.realtime = realtime
        if not self.path.is_file():
            raise SourceUnavailable(f"replay file not found: {self.path}")
        self._snap: frozenset[int] | None = None

    def __iter__(self) -> Iterator[Event]:
        prev = None
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("type") == "proc_snapshot":
                    self._snap = frozenset(d["pids"])
                    continue
                ev = Event.from_dict(d)
                if self.realtime and prev is not None:
                    time.sleep(max(0.0, min((ev.ts_ns - prev) / 1e9, 1.0)))
                prev = ev.ts_ns
                yield ev

    def proc_snapshot(self):
        return self._snap


class MockSource(EventSource):
    """Deterministic synthetic traffic: concurrent benign sessions with attacks injected."""
    name = "mock"

    def __init__(self, seed: int = 0, attack_rate: float = 0.08, max_events: int | None = None,
                 batch_sessions: int = 24):
        self.rng = random.Random(seed)
        self.attack_rate = attack_rate
        self.max_events = max_events
        self.batch = batch_sessions
        self._visible: set[int] = set()
        self._next_pid = 20_000
        self._t = 1_000_000_000

    def proc_snapshot(self):
        return frozenset(self._visible)

    def _make_batch(self):
        sessions = []
        for _ in range(self.batch):
            kind = self.rng.choice(ATTACK_KINDS) if self.rng.random() < self.attack_rate else random_benign_kind(self.rng)
            start = self._t + int(self.rng.uniform(0, 2e9))
            s = make_session(kind, self.rng, self._next_pid, start)
            self._next_pid += 200
            sessions.append(s)
        self._t += int(3e9)
        return sessions

    def __iter__(self) -> Iterator[Event]:
        n = 0
        while True:
            sessions = self._make_batch()
            for s in sessions:
                self._visible |= (s.pids - s.hidden_pids)
            merged = heapq.merge(*[s.events for s in sessions], key=lambda e: e.ts_ns)
            for ev in merged:
                yield ev
                n += 1
                if self.max_events is not None and n >= self.max_events:
                    return
            self._t = max(self._t, ev.ts_ns)


def live_capable() -> tuple[bool, str]:
    """Cheap pre-flight check: Linux, loader built, BPF-capable privileges, tracefs present."""
    if not sys.platform.startswith("linux"):
        return False, "not linux"
    if not (DEFAULT_OBJ.is_file() and DEFAULT_LOADER.is_file()):
        return False, "bpf objects not built (run `make bpf`)"
    try:
        capeff = int(next(l for l in open("/proc/self/status") if l.startswith("CapEff")).split()[1], 16)
    except (OSError, StopIteration, ValueError):
        capeff = 0
    if not (capeff >> 39 & 1 or capeff >> 21 & 1 or os.geteuid() == 0):
        return False, "missing CAP_BPF/CAP_SYS_ADMIN"
    if not (Path("/sys/kernel/tracing").is_dir() or Path("/sys/kernel/debug/tracing").is_dir()):
        return False, "tracefs unavailable"
    return True, "ok"


class LiveSource(EventSource):
    name = "live"

    def __init__(self, loader: str | os.PathLike = DEFAULT_LOADER, obj: str | os.PathLike = DEFAULT_OBJ):
        ok, why = live_capable()
        if not ok:
            raise SourceUnavailable(why)
        self.proc = subprocess.Popen([str(loader), str(obj)], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        time.sleep(0.4)
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() or "").strip().splitlines()[-1:] or ["loader exited"]
            raise SourceUnavailable(err[0])
        self._snap: frozenset[int] = read_proc_pids()
        self._snap_t = time.monotonic()

    def proc_snapshot(self):
        if time.monotonic() - self._snap_t > 1.0:
            self._snap, self._snap_t = read_proc_pids(), time.monotonic()
        return self._snap

    def __iter__(self) -> Iterator[Event]:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                yield Event.from_json(line)
            except (ValueError, KeyError):
                continue

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def auto_source(mode: str = "auto", replay: str | None = None, seed: int = 0,
                attack_rate: float = 0.08, max_events: int | None = None) -> tuple[EventSource, str]:
    """Return (source, note). ``auto`` tries live and silently falls back to mock."""
    if mode == "replay":
        if not replay:
            raise SourceUnavailable("--replay FILE required for replay mode")
        return ReplaySource(replay), f"replaying {replay}"
    if mode == "mock":
        return MockSource(seed, attack_rate, max_events), "mock traffic"
    if mode in ("live", "auto"):
        try:
            return LiveSource(), "live eBPF"
        except SourceUnavailable as e:
            if mode == "live":
                raise
            return MockSource(seed, attack_rate, max_events), f"live unavailable ({e}); using mock"
    raise ValueError(f"unknown mode {mode}")


def record(source: EventSource, path: str | os.PathLike, max_events: int) -> int:
    """Write events (and periodic /proc snapshots) to JSONL for later replay."""
    n, last = 0, None
    with open(path, "w") as f:
        for ev in source:
            snap = source.proc_snapshot()
            if snap is not None and snap != last and n % 64 == 0:
                f.write(json.dumps({"type": "proc_snapshot", "pids": sorted(snap)}) + "\n")
                last = snap
            f.write(ev.to_json() + "\n")
            n += 1
            if n >= max_events:
                break
    return n
