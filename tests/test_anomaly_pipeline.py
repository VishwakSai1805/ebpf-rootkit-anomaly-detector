"""Feature extractor + Isolation Forest/Autoencoder detector + streaming pipeline."""
import random
import time

import numpy as np
import pytest

from collector.events import Event, FLAG_LD_PRELOAD_ENV, PTRACE_ATTACH, PTRACE_POKETEXT
from collector.scenarios import ATTACK_KINDS, BENIGN_KINDS, generate_sessions, make_session
from collector.sources import MockSource
from ml.dataset import benign_training_matrix, session_windows
from ml.detector import IF_COLS, AnomalyDetector, categorize
from ml.features import (FEATURE_DIM, SC_OFF, S, HiddenPidChecker, WindowFeaturizer, extract, tokenize,
                         feature_names, T)
from ml.pipeline import StreamingDetector


@pytest.fixture(scope="module")
def model():
    return AnomalyDetector(seed=0).fit(benign_training_matrix(500, seed=1234))


def ev(type, i=0, pid=10, **kw):
    return Event(ts_ns=i * 1_000_000, pid=pid, type=type, **kw)


# ---------------------------------------------------------------- features
def test_feature_dim_and_names():
    assert len(feature_names()) == FEATURE_DIM
    x = extract([ev("execve", 0, data="/bin/ls")] * 5)
    assert x.shape == (FEATURE_DIM,) and x.dtype == np.float32 and np.isfinite(x).all()


def test_ngram_frequencies_sum_to_one():
    w = [ev("fork", 0, arg0=11), ev("execve", 1, data="/bin/ls"), ev("fork", 2, arg0=12), ev("execve", 3, data="/bin/ls")]
    x = extract(w)
    assert x[:20].sum() == pytest.approx(1.0)
    assert x[20:20 + 400].sum() == pytest.approx(1.0)


def test_tokenizer_classes():
    assert tokenize(ev("execve", data="/tmp/x")) == T["EXEC_TMP"]
    assert tokenize(ev("execve", data="/bin/bash")) == T["EXEC_SHELL"]
    assert tokenize(ev("execve", data="/bin/ls", flags=FLAG_LD_PRELOAD_ENV)) == T["EXEC_PRELOAD"]
    assert tokenize(ev("connect", data="10.0.0.1")) == T["CONN_LOCAL"]
    assert tokenize(ev("connect", data="8.8.8.8")) == T["CONN_EXT"]
    assert tokenize(ev("dup", arg1=1)) == T["DUP_STDIO"] and tokenize(ev("dup", arg1=7)) == T["DUP_OTHER"]
    assert tokenize(ev("ptrace", arg0=PTRACE_POKETEXT)) == T["PT_POKE"]
    assert tokenize(ev("commit_creds", uid=1000, euid=0)) == T["CREDS_ROOT"]
    assert tokenize(ev("commit_creds", uid=0, euid=1000)) == T["CREDS"]


def test_uid_transition_flags():
    sudo = [ev("execve", 0, data="/usr/bin/sudo", uid=1000, euid=1000),
            ev("commit_creds", 1, uid=1000, euid=0, arg0=0)]
    expl = extract(sudo)[SC_OFF:]
    assert expl[S["root_transition"]] == 1 and expl[S["unexplained_root"]] == 0
    exploit = [ev("execve", 0, data="/tmp/x", uid=1000, euid=1000), ev("commit_creds", 1, uid=1000, euid=0)]
    bad = extract(exploit)[SC_OFF:]
    assert bad[S["unexplained_root"]] == 1 and bad[S["exec_tmp"]] > 0


def test_reverse_shell_chain_needs_socket_fd_dups():
    chain = [ev("connect", 0, arg0=3, data="8.8.8.8", arg1=4444), ev("dup", 1, arg0=3, arg1=0),
             ev("dup", 2, arg0=3, arg1=1), ev("execve", 3, data="/bin/sh")]
    assert extract(chain)[SC_OFF + S["revshell_chain"]] == 1
    pipeline = [ev("connect", 0, arg0=3, data="8.8.8.8"), ev("dup", 1, arg0=9, arg1=0),
                ev("dup", 2, arg0=9, arg1=1), ev("execve", 3, data="/bin/sh")]
    assert extract(pipeline)[SC_OFF + S["revshell_chain"]] == 0


def test_ptrace_foreign_vs_child():
    own = [ev("fork", 0, arg0=55), ev("ptrace", 1, arg0=PTRACE_ATTACH, arg1=55)]
    foreign = [ev("ptrace", 1, arg0=PTRACE_ATTACH, arg1=4242)]
    assert extract(own)[SC_OFF + S["ptrace_foreign"]] == 0
    assert extract(foreign)[SC_OFF + S["ptrace_foreign"]] == 1


def test_hidden_pid_crosscheck():
    chk = HiddenPidChecker(grace_ns=1_000_000)
    chk.observe(ev("execve", 0, pid=100))
    chk.observe(ev("fork", 1, pid=100, arg0=101))
    chk.observe(ev("execve", 2, pid=666))
    chk.observe(ev("exit", 3, pid=101))
    proc = frozenset({100})
    assert chk.hidden({100, 101, 666}, proc, 10_000_000) == {666}       # 101 exited, 100 visible
    assert chk.hidden({666}, proc, 500_000) == set()                    # inside grace window
    assert chk.hidden({666}, None, 10_000_000) == set()                 # no snapshot -> no verdict


def test_hidden_pid_feature_end_to_end():
    import random
    s = make_session("hidden_pid", random.Random(2), 500, 0)
    last = [x for x, *_ in session_windows(s)][-1]
    assert last[SC_OFF + S["hidden_pids"]] > 0
    b = make_session("shell", random.Random(2), 500, 0)
    assert all(x[SC_OFF + S["hidden_pids"]] == 0 for x, *_ in session_windows(b))


# ---------------------------------------------------------------- model
def test_forest_export_matches_sklearn(model):
    X = benign_training_matrix(40, seed=7)[:200]
    ref = -model._sk_forest.score_samples(X[:, IF_COLS])
    assert np.allclose(model.if_scores(X), ref, atol=1e-4)


def test_save_load_roundtrip(model, tmp_path):
    p = tmp_path / "m.npz"
    model.save(p)
    m2 = AnomalyDetector.load(p)
    X = benign_training_matrix(20, seed=8)[:100]
    assert np.allclose(model.score_batch(X)[0], m2.score_batch(X)[0])


def _session_hit(model, s):
    return any(model.detect(x).anomaly and lab != "benign" for x, lab, *_ in session_windows(s))


@pytest.fixture(scope="module")
def corpus():
    return generate_sessions(150, 30, seed=4242)


@pytest.mark.parametrize("kind,min_recall", [("privesc", 0.9), ("reverse_shell", 0.85), ("ptrace_injection", 0.9),
                                              ("hidden_pid", 0.9), ("ld_preload", 0.65)])
def test_attack_detection(model, corpus, kind, min_recall):
    ss = [s for s in corpus if s.label == kind]
    recall = np.mean([_session_hit(model, s) for s in ss])
    assert recall >= min_recall, f"{kind}: recall {recall:.2f}"


def test_benign_false_positive_rate_bounded(model, corpus):
    ss = [s for s in corpus if not s.is_attack]
    fp = np.mean([any(model.detect(x).anomaly for x, *_ in session_windows(s)) for s in ss])
    assert fp <= 0.12, f"session FPR {fp:.2f}"
    wins = [x for s in ss for x, *_ in session_windows(s, stride=3)]
    assert (model.score_batch(np.stack(wins))[0] >= 1).mean() <= 0.02


def test_categories_are_attributed(model):
    rng = random.Random(11)
    want = {"privesc": "privilege_escalation", "reverse_shell": "reverse_shell",
            "ptrace_injection": "ptrace_injection", "hidden_pid": "hidden_process"}
    for kind, cat in want.items():
        got = set()
        for i in range(10):
            s = make_session(kind, rng, 1000 + i * 200, 0)
            for x, *_ in session_windows(s):
                d = model.detect(x)
                if d.anomaly:
                    got.update(d.categories)
        assert cat in got, (kind, got)


def test_latency_under_one_millisecond(model):
    xs = benign_training_matrix(20, seed=9)[:300]
    for x in xs[:20]:
        model.detect(x)  # warm-up
    lat = []
    for x in xs:
        t = time.perf_counter()
        model.detect(x)
        lat.append(time.perf_counter() - t)
    assert np.median(lat) < 1e-3, f"median {np.median(lat) * 1e6:.0f}us"
    # features + inference end to end
    wf = WindowFeaturizer()
    src = MockSource(seed=5, max_events=600)
    e2e = []
    for e in src:
        t = time.perf_counter()
        r = wf.push(e, src.proc_snapshot())
        if r:
            model.detect(r[0])
            e2e.append(time.perf_counter() - t)
    assert np.median(e2e) < 1e-3, f"end-to-end median {np.median(e2e) * 1e6:.0f}us"


def test_streaming_pipeline_emits_alerts_with_cooldown(model):
    sd = StreamingDetector(model)
    src = MockSource(seed=21, attack_rate=0.5, max_events=6000)
    alerts = [a for e in src if (a := sd.process(e, src.proc_snapshot()))]
    assert alerts and sd.windows_scored > 1000
    keys = [(a.pid, a.categories[0]) for a in alerts]
    assert len(keys) == len(set(keys)) or all(  # cooldown: repeats only >5s apart
        b.ts_ns - a.ts_ns >= 5_000_000_000 for i, a in enumerate(alerts) for b in alerts[i + 1:]
        if (a.pid, a.categories[0]) == (b.pid, b.categories[0]))
    hit = sum(1 for a in alerts if a.truth)
    assert hit / len(alerts) > 0.7  # most alerts correspond to injected attacks
    assert '"categories"' in alerts[0].to_json()
