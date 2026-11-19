import json
import shutil
import subprocess
from pathlib import Path

import pytest

from collector.events import Event, EVT_NAMES
from collector.scenarios import ATTACK_KINDS, BENIGN_KINDS, generate_sessions, make_session
from collector.sources import (LiveSource, MockSource, ReplaySource, SourceUnavailable,
                               auto_source, live_capable, record)

ROOT = Path(__file__).resolve().parent.parent


def test_event_roundtrip_and_numeric_type():
    e = Event(ts_ns=5, pid=7, type="ptrace", arg0=16, arg1=99, comm="x", label="a")
    assert Event.from_json(e.to_json()) == e
    assert Event.from_dict({"ts_ns": 1, "pid": 2, "type": 1}).type == "execve"
    assert set(EVT_NAMES.values()) >= {"execve", "ptrace", "commit_creds", "connect"}


def test_mock_is_deterministic():
    a = [e.to_json() for e in MockSource(seed=3, max_events=500)]
    b = [e.to_json() for e in MockSource(seed=3, max_events=500)]
    c = [e.to_json() for e in MockSource(seed=4, max_events=500)]
    assert a == b and a != c and len(a) == 500


def test_mock_time_ordered_and_hidden_pids_absent_from_proc():
    src = MockSource(seed=1, attack_rate=1.0, max_events=2000)
    last, seen = 0, set()
    for e in src:
        assert e.ts_ns >= last
        last = e.ts_ns
        if e.label == "hidden_pid":
            seen.add(e.pid)
    assert seen and not (seen & src.proc_snapshot())


@pytest.mark.parametrize("kind", ATTACK_KINDS + BENIGN_KINDS)
def test_every_scenario_generates(kind):
    import random
    s = make_session(kind, random.Random(0), 1000, 0)
    assert s.events and s.is_attack == (kind in ATTACK_KINDS)
    if s.is_attack:
        assert any(e.label == kind for e in s.events)


def test_corpus_counts():
    ss = generate_sessions(10, 3, seed=1)
    assert sum(not s.is_attack for s in ss) == 10 and sum(s.is_attack for s in ss) == 3 * len(ATTACK_KINDS)


def test_record_then_replay(tmp_path):
    p = tmp_path / "cap.jsonl"
    n = record(MockSource(seed=2, attack_rate=0.5), p, 300)
    assert n == 300
    rs = ReplaySource(p)
    evs = list(rs)
    assert len(evs) == 300 and rs.proc_snapshot() is not None
    first = next(json.loads(l) for l in p.read_text().splitlines() if '"proc_snapshot"' not in l)
    assert evs[0].to_dict() == first


def test_replay_missing_file():
    with pytest.raises(SourceUnavailable):
        ReplaySource("/nonexistent.jsonl")


def test_auto_falls_back_to_mock_without_kernel_caps():
    ok, _ = live_capable()
    src, note = auto_source("auto", max_events=10)
    try:
        if not ok:
            assert src.name == "mock" and "mock" in note
        assert src.name in ("mock", "live")
    finally:
        src.close()


def test_live_mode_raises_cleanly_when_unavailable():
    # Either live works (privileged host) or it must raise SourceUnavailable, never crash.
    try:
        src, _ = auto_source("live")
    except SourceUnavailable:
        return
    src.close()
    assert src.name == "live"


@pytest.mark.skipif(not (shutil.which("clang") and Path("/usr/include/bpf/bpf_helpers.h").exists()),
                    reason="clang/libbpf headers not installed")
def test_bpf_program_compiles(tmp_path):
    out = tmp_path / "probes.bpf.o"
    r = subprocess.run(["make", "-s", "-C", str(ROOT), f"BUILD={tmp_path}", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 1000


def test_probe_sources_cover_required_hooks():
    src = (ROOT / "bpf" / "probes.bpf.c").read_text()
    for hook in ("sys_enter_execve", "sys_enter_ptrace", "kprobe/commit_creds", "sys_enter_connect",
                 "sys_enter_socket", "sys_enter_dup2"):
        assert hook in src
