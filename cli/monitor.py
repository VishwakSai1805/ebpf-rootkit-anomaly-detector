"""Real-time CLI dashboard + JSONL alert exporter.

  python -m cli.monitor --source mock --duration 10            # ANSI dashboard
  python -m cli.monitor --source auto --plain --json-out alerts.jsonl
  python -m cli.monitor --source replay --replay capture.jsonl --plain
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

from collector.sources import SourceUnavailable, auto_source, record
from ml.detector import AnomalyDetector
from ml.pipeline import Alert, StreamingDetector, load_or_train

SEV_COLOR = {"critical": "\x1b[1;37;41m", "high": "\x1b[1;31m", "medium": "\x1b[33m"}
RESET, BOLD, DIM, CYAN, GREEN = "\x1b[0m", "\x1b[1m", "\x1b[2m", "\x1b[36m", "\x1b[32m"


class Stats:
    def __init__(self, source_note: str):
        self.note = source_note
        self.t0 = time.monotonic()
        self.events = 0
        self.by_type: collections.Counter = collections.Counter()
        self.by_cat: collections.Counter = collections.Counter()
        self.alerts = 0
        self.recent: collections.deque[Alert] = collections.deque(maxlen=12)
        self.lat_us: collections.deque[float] = collections.deque(maxlen=2000)
        self.true_hits = 0

    def add_alert(self, a: Alert) -> None:
        self.alerts += 1
        self.recent.append(a)
        self.true_hits += bool(a.truth)
        for c in a.categories:
            self.by_cat[c] += 1


def _bar(v: int, mx: int, width: int = 24) -> str:
    n = 0 if mx == 0 else round(width * v / mx)
    return "█" * n + "·" * (width - n)


def render(stats: Stats, windows: int, color: bool = True, width: int = 100, now: float | None = None) -> str:
    """Pure function: build one dashboard frame (testable without a TTY)."""
    c = (lambda code, s: f"{code}{s}{RESET}") if color else (lambda code, s: s)
    el = max((now if now is not None else time.monotonic()) - stats.t0, 1e-9)
    lat = sorted(stats.lat_us)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p99 = lat[int(len(lat) * 0.99)] if lat else 0.0
    out = [c(BOLD, "eBPF Rootkit/Anomaly Detector") + c(DIM, f"  [{stats.note}]"),
           f"uptime {el:6.1f}s  events {stats.events:>8}  ({stats.events / el:7.0f}/s)  windows scored {windows:>8}  "
           f"alerts {c(BOLD, str(stats.alerts))}",
           f"inference latency p50 {p50:6.0f}µs  p99 {p99:6.0f}µs  (budget 1000µs)", ""]
    out.append(c(CYAN, "syscall mix"))
    mx = max(stats.by_type.values(), default=0)
    for t, n in stats.by_type.most_common(9):
        out.append(f"  {t:<13}{_bar(n, mx)} {n}")
    out.append("")
    out.append(c(CYAN, "alerts by category"))
    if stats.by_cat:
        mx = max(stats.by_cat.values())
        for t, n in stats.by_cat.most_common():
            out.append(f"  {t:<22}{_bar(n, mx, 20)} {n}")
    else:
        out.append(c(DIM, "  none"))
    out += ["", c(CYAN, "recent alerts"),
            c(DIM, f"  {'sev':<9}{'pid':>7} {'comm':<14}{'score':>6}  categories / recent calls")]
    for a in reversed(stats.recent):
        line = f"  {a.severity:<9}{a.pid:>7} {a.comm[:13]:<14}{a.score:>6.2f}  {','.join(a.categories)}  {c(DIM, ' '.join(a.recent[-3:]))}"
        out.append(c(SEV_COLOR[a.severity], line[:width]) if a.severity == "critical" else
                   (c(SEV_COLOR[a.severity], line[:20]) + line[20:])[:width + 20])
    return "\n".join(out)


class AlertExporter:
    """Appends one JSON object per alert; ``-`` means stdout."""

    def __init__(self, path: str | None):
        self.fh = None
        if path:
            self.fh = sys.stdout if path == "-" else open(path, "a", buffering=1)
            self.is_stdout = path == "-"

    def write(self, a: Alert) -> None:
        if self.fh:
            self.fh.write(a.to_json() + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh and not self.is_stdout:
            self.fh.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="monitor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "live", "mock", "replay"], default="auto")
    ap.add_argument("--replay", help="JSONL capture for --source replay")
    ap.add_argument("--record", metavar="FILE", help="record the event stream to JSONL (see --max-events) and exit")
    ap.add_argument("--model", help="model .npz (trained on demand if missing)")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--json-out", metavar="FILE", help="append alerts as JSONL ('-' = stdout)")
    ap.add_argument("--plain", action="store_true", help="line-oriented log instead of the dashboard")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl-C/EOF)")
    ap.add_argument("--max-events", type=int, default=0, help="stop after N events (0 = unlimited)")
    ap.add_argument("--rate", type=float, default=2000, help="throttle mock/replay to N events/s (0 = as fast as possible)")
    ap.add_argument("--attack-rate", type=float, default=0.08, help="mock: fraction of sessions that are attacks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", type=float, default=0.25, help="dashboard refresh seconds")
    ap.add_argument("--no-color", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        src, note = auto_source(a.source, a.replay, a.seed, a.attack_rate, a.max_events or None)
    except SourceUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if a.record:
        n = record(src, a.record, a.max_events or 5000)
        src.close()
        print(f"recorded {n} events to {a.record}")
        return 0

    model: AnomalyDetector = load_or_train(a.model, retrain=a.retrain)
    det = StreamingDetector(model)
    det.keep_latencies = True
    stats = Stats(note)
    exporter = AlertExporter(a.json_out)
    ui = sys.stderr if a.json_out == "-" else sys.stdout
    color = not a.no_color and not os.environ.get("NO_COLOR") and ui.isatty()
    dashboard = not a.plain
    t0 = last_draw = time.monotonic()
    try:
        if dashboard:
            ui.write("\x1b[?25l\x1b[2J")
        for ev in src:
            stats.events += 1
            stats.by_type[ev.type] += 1
            alert = det.process(ev, src.proc_snapshot())
            if det.latencies_ns:
                stats.lat_us.extend(l / 1e3 for l in det.latencies_ns)
                det.latencies_ns.clear()
            if alert:
                stats.add_alert(alert)
                exporter.write(alert)
                if not dashboard:
                    ui.write(f"[{alert.severity.upper():8}] pid={alert.pid} comm={alert.comm} score={alert.score} "
                             f"cat={','.join(alert.categories)} recent={' '.join(alert.recent[-3:])}\n")
            now = time.monotonic()
            if dashboard and now - last_draw >= a.refresh:
                ui.write("\x1b[H\x1b[J" + render(stats, det.windows_scored, color) + "\n")
                ui.flush()
                last_draw = now
            if a.max_events and stats.events >= a.max_events:
                break
            if a.duration and now - t0 >= a.duration:
                break
            if a.rate > 0 and src.name != "live":
                ahead = stats.events / a.rate - (now - t0)
                if ahead > 0.002:
                    time.sleep(ahead)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        exporter.close()
        if dashboard:
            ui.write("\x1b[H\x1b[J" + render(stats, det.windows_scored, color) + "\x1b[?25h\n")
        el = time.monotonic() - t0
        ui.write(f"-- {stats.events} events, {det.windows_scored} windows, {stats.alerts} alerts in {el:.1f}s "
                 f"({stats.events / max(el, 1e-9):.0f} ev/s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
