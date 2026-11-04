"""Behavioural feature extraction: syscall n-grams, UID-transition flags, hidden-PID cross-check.

Online and offline paths share ``WindowFeaturizer`` so training windows match production windows.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Iterable, Sequence

import numpy as np

from collector.events import (
    Event, FLAG_LD_PRELOAD_ENV, PTRACE_ATTACH, PTRACE_POKEDATA, PTRACE_POKETEXT, PTRACE_POKEUSER,
    PTRACE_SEIZE, PTRACE_SETFPREGS, PTRACE_SETREGS,
)

TOKENS = ["EXEC", "EXEC_SHELL", "EXEC_TMP", "EXEC_PRELOAD", "PT_ATTACH", "PT_POKE", "PT_REGS", "PT_OTHER",
          "CREDS", "CREDS_ROOT", "CONN_EXT", "CONN_LOCAL", "SOCKET", "DUP_STDIO", "DUP_OTHER",
          "OPEN_PRELOAD", "OPEN", "FORK", "EXIT", "OTHER"]
T = {n: i for i, n in enumerate(TOKENS)}
NT = len(TOKENS)
TRI_BUCKETS = 64
SCALARS = ["fill", "root_transition", "unexplained_root", "euid_mismatch", "exec_tmp", "ptrace_attach",
           "ptrace_poke", "ptrace_regs", "ptrace_foreign", "preload_env", "preload_open", "ext_connect",
           "dup_stdio", "sock_dup", "revshell_chain", "hidden_pids", "shell_exec"]
S = {n: i for i, n in enumerate(SCALARS)}
N_UNI, N_BI, N_TRI, N_SC = NT, NT * NT, TRI_BUCKETS, len(SCALARS)
FEATURE_DIM = N_UNI + N_BI + N_TRI + N_SC
SC_OFF = N_UNI + N_BI + N_TRI

_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ash", "ksh", "fish"})
_SETUID = frozenset({"sudo", "su", "passwd", "ping", "mount", "umount", "pkexec", "newgrp", "chsh", "doas"})
_TMP = ("/tmp/", "/dev/shm/", "/var/tmp/")
_POKES = {PTRACE_POKETEXT, PTRACE_POKEDATA, PTRACE_POKEUSER}
_REGS = {PTRACE_SETREGS, PTRACE_SETFPREGS}


def feature_names() -> list[str]:
    return ([f"uni:{t}" for t in TOKENS]
            + [f"bi:{a}>{b}" for a in TOKENS for b in TOKENS]
            + [f"tri#{i}" for i in range(TRI_BUCKETS)]
            + [f"sc:{s}" for s in SCALARS])


def _is_external(ip: str) -> bool:
    if not ip or ip.startswith(("127.", "10.", "192.168.", "169.254.", "0.")):
        return False  # private/loopback (or IPv6, whose address the probe does not capture)
    if ip.startswith("172."):
        try:
            return not 16 <= int(ip.split(".")[1]) <= 31
        except ValueError:
            return True
    return True


def tokenize(ev: Event) -> int:
    t = ev.type
    if t == "execve":
        p = ev.data
        if ev.flags & FLAG_LD_PRELOAD_ENV:
            return T["EXEC_PRELOAD"]
        if p.startswith(_TMP):
            return T["EXEC_TMP"]
        if p.rsplit("/", 1)[-1] in _SHELLS:
            return T["EXEC_SHELL"]
        return T["EXEC"]
    if t == "ptrace":
        r = ev.arg0
        if r in (PTRACE_ATTACH, PTRACE_SEIZE):
            return T["PT_ATTACH"]
        if r in _POKES:
            return T["PT_POKE"]
        if r in _REGS:
            return T["PT_REGS"]
        return T["PT_OTHER"]
    if t == "commit_creds":
        return T["CREDS_ROOT"] if (ev.euid == 0 and ev.uid != 0) else T["CREDS"]
    if t == "connect":
        return T["CONN_EXT"] if _is_external(ev.data) else T["CONN_LOCAL"]
    if t == "socket":
        return T["SOCKET"]
    if t == "dup":
        return T["DUP_STDIO"] if ev.arg1 in (0, 1, 2) else T["DUP_OTHER"]
    if t == "open":
        return T["OPEN_PRELOAD"] if ev.data.endswith("ld.so.preload") else T["OPEN"]
    if t == "fork":
        return T["FORK"]
    if t == "exit":
        return T["EXIT"]
    return T["OTHER"]


def extract(window: Sequence[Event], hidden_pids: int = 0, max_len: int = 32) -> np.ndarray:
    """Feature vector for one window (float32, length FEATURE_DIM)."""
    n = len(window)
    toks = np.fromiter((tokenize(e) for e in window), dtype=np.int64, count=n)
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    if n:
        f[:N_UNI] = np.bincount(toks, minlength=NT) / n
    if n >= 2:
        bi = toks[:-1] * NT + toks[1:]
        f[N_UNI:N_UNI + N_BI] = np.bincount(bi, minlength=N_BI) / (n - 1)
    if n >= 3:
        tri = (toks[:-2] * 31 + toks[1:-1] * 7 + toks[2:]) % TRI_BUCKETS
        f[N_UNI + N_BI:SC_OFF] = np.bincount(tri, minlength=TRI_BUCKETS) / (n - 2)

    children = {e.arg0 for e in window if e.type == "fork"}
    last_setuid = -99
    unexplained = root = mismatch = foreign = 0
    chain = False
    sock_fds: set[int] = set()
    sock_dups = 0
    for i, e in enumerate(window):
        k = toks[i]
        if e.uid != 0 and e.euid == 0:
            mismatch += 1
        if k == T["EXEC"] and e.data.rsplit("/", 1)[-1] in _SETUID:
            last_setuid = i
        elif k == T["CREDS_ROOT"]:
            root += 1
            if i - last_setuid > 3:
                unexplained += 1
        elif k == T["PT_ATTACH"] and e.arg1 not in children:
            foreign = 1
        if k == T["CONN_EXT"]:
            sock_fds.add(e.arg0)
        elif k == T["DUP_STDIO"] and e.arg0 in sock_fds:  # connected socket fd duplicated onto stdio
            sock_dups += 1
        elif k == T["EXEC_SHELL"] and sock_dups >= 2:
            chain = True
    cnt = np.bincount(toks, minlength=NT) if n else np.zeros(NT, dtype=np.int64)
    v = f[SC_OFF:]
    v[S["fill"]] = n / max_len
    v[S["root_transition"]] = min(root, 3)
    v[S["unexplained_root"]] = min(unexplained, 3)
    v[S["euid_mismatch"]] = mismatch / max(n, 1)
    v[S["exec_tmp"]] = np.log1p(cnt[T["EXEC_TMP"]])
    v[S["ptrace_attach"]] = np.log1p(cnt[T["PT_ATTACH"]])
    v[S["ptrace_poke"]] = np.log1p(cnt[T["PT_POKE"]])
    v[S["ptrace_regs"]] = np.log1p(cnt[T["PT_REGS"]])
    v[S["ptrace_foreign"]] = foreign
    v[S["preload_env"]] = np.log1p(cnt[T["EXEC_PRELOAD"]])
    v[S["preload_open"]] = np.log1p(cnt[T["OPEN_PRELOAD"]])
    v[S["ext_connect"]] = np.log1p(cnt[T["CONN_EXT"]])
    v[S["dup_stdio"]] = np.log1p(cnt[T["DUP_STDIO"]])
    v[S["sock_dup"]] = np.log1p(sock_dups)
    v[S["revshell_chain"]] = 1.0 if chain else 0.0
    v[S["hidden_pids"]] = np.log1p(hidden_pids)
    v[S["shell_exec"]] = np.log1p(cnt[T["EXEC_SHELL"]])
    return f


class HiddenPidChecker:
    """Cross-checks PIDs the kernel reports (events, fork children) against a /proc snapshot.

    A PID is *hidden* if the kernel saw it alive (no exit event) for longer than ``grace_ns`` yet
    it is absent from /proc. The grace period absorbs races between event time and snapshot time.
    """

    def __init__(self, grace_ns: int = 1_000_000_000, max_tracked: int = 65536):
        self.grace_ns = grace_ns
        self.max_tracked = max_tracked
        self.first_seen: OrderedDict[int, int] = OrderedDict()

    def observe(self, ev: Event) -> None:
        if ev.type == "exit":
            self.first_seen.pop(ev.pid, None)
            return
        if ev.pid not in self.first_seen:
            self.first_seen[ev.pid] = ev.ts_ns
        if ev.type == "fork" and ev.arg0 not in self.first_seen:
            self.first_seen[ev.arg0] = ev.ts_ns
        while len(self.first_seen) > self.max_tracked:
            self.first_seen.popitem(last=False)

    def hidden(self, pids: Iterable[int], proc: frozenset[int] | None, now_ns: int) -> set[int]:
        if proc is None:
            return set()
        fs = self.first_seen
        return {p for p in pids if p in fs and p not in proc and now_ns - fs[p] > self.grace_ns}


class WindowFeaturizer:
    """Per-PID sliding windows -> feature vectors. ``push`` returns (features, window) or None."""

    def __init__(self, window: int = 32, min_events: int = 8, grace_ns: int = 1_000_000_000,
                 max_pids: int = 8192):
        self.window, self.min_events, self.max_pids = window, min_events, max_pids
        self.windows: OrderedDict[int, deque[Event]] = OrderedDict()
        self.checker = HiddenPidChecker(grace_ns)

    def push(self, ev: Event, proc: frozenset[int] | None = None):
        self.checker.observe(ev)
        if ev.type == "exit":
            self.windows.pop(ev.pid, None)
            return None
        w = self.windows.get(ev.pid)
        if w is None:
            w = self.windows[ev.pid] = deque(maxlen=self.window)
            while len(self.windows) > self.max_pids:
                self.windows.popitem(last=False)
        else:
            self.windows.move_to_end(ev.pid)
        w.append(ev)
        if len(w) < self.min_events:
            return None
        pids = {ev.pid}
        pids.update(e.arg0 for e in w if e.type == "fork")
        hidden = self.checker.hidden(pids, proc, ev.ts_ns)
        return extract(w, len(hidden), self.window), w
