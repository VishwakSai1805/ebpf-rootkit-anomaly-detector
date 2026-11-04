"""Deterministic synthetic traffic: labelled benign workloads and attack chains.

Every generator is pure given its ``random.Random``. A *session* is the event stream of one
process (pid) plus the child-exit events its forks imply; attack sessions start with benign
shell activity so that sliding windows contain a realistic mixture.
Simplification (documented in SPEC.md): fork+exec of a child is attributed to the parent's pid.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .events import (
    Event, FLAG_LD_PRELOAD_ENV, PTRACE_ATTACH, PTRACE_CONT, PTRACE_DETACH, PTRACE_GETREGS,
    PTRACE_PEEKTEXT, PTRACE_POKETEXT, PTRACE_SEIZE, PTRACE_SETREGS, PTRACE_TRACEME,
)

BENIGN_KINDS = ("shell", "sudo", "root_admin", "daemon", "build", "debugger", "preload_dev", "ssh_login")
BENIGN_WEIGHTS = (0.40, 0.10, 0.08, 0.17, 0.12, 0.04, 0.02, 0.07)
ATTACK_KINDS = ("privesc", "reverse_shell", "ld_preload", "ptrace_injection", "hidden_pid")
BENIGN = "benign"

_CMDS = ["ls", "cat", "grep", "git", "python3", "vim", "make", "curl", "ssh", "ps", "top", "find",
         "tar", "less", "cp", "mv", "rm", "awk", "sed", "docker", "pip", "node", "wget", "htop"]
_NET_CMDS = {"curl", "git", "ssh", "wget", "pip", "docker", "node"}
_SHELLS = ["/bin/sh", "/bin/bash", "/usr/bin/zsh", "/usr/bin/dash"]
_ATTACKER_PORTS = [4444, 1337, 9001, 31337, 5555, 8081, 6666]


@dataclass
class Session:
    kind: str
    label: str
    events: list[Event]
    pids: set[int] = field(default_factory=set)
    hidden_pids: set[int] = field(default_factory=set)

    @property
    def is_attack(self) -> bool:
        return self.label != BENIGN


class _Gen:
    def __init__(self, rng: random.Random, pid: int, t0: int, uid: int = 1000, comm: str = "bash"):
        self.rng, self.pid, self.t, self.uid, self.euid, self.comm = rng, pid, t0, uid, uid, comm
        self.ppid = 1
        self.events: list[Event] = []
        self.children: list[Event] = []  # deferred child exits
        self.pids = {pid}
        self._next_child = pid + 1

    def wait(self, lo_ms: float = 0.2, hi_ms: float = 40.0) -> None:
        self.t += int(self.rng.uniform(lo_ms, hi_ms) * 1e6)

    def emit(self, type: str, label: str = BENIGN, wait=(0.05, 3.0), **kw) -> Event:
        self.wait(*wait)
        kw.setdefault("uid", self.uid)
        kw.setdefault("euid", self.euid)
        ev = Event(ts_ns=self.t, pid=self.pid, tid=self.pid, ppid=self.ppid, type=type,
                   comm=kw.pop("comm", self.comm), label=label, **kw)
        self.events.append(ev)
        return ev

    def fork(self, label: str = BENIGN) -> int:
        c = self._next_child
        self._next_child += 1
        self.pids.add(c)
        self.emit("fork", label, arg0=c)
        self.children.append(Event(ts_ns=self.t + int(self.rng.uniform(5, 400) * 1e6), pid=c, tid=c,
                                   ppid=self.pid, type="exit", uid=self.uid, euid=self.euid,
                                   comm="child", label=BENIGN))
        return c

    def execve(self, path: str, label: str = BENIGN, flags: int = 0, **kw) -> Event:
        return self.emit("execve", label, data=path, flags=flags, **kw)

    def connect(self, ip: str, port: int, label: str = BENIGN, fd: int = 3) -> None:
        self.emit("socket", label, arg0=2, arg1=1)
        self.emit("connect", label, arg0=fd, arg1=port, data=ip, wait=(0.1, 20))

    def finish(self, kind: str, label: str, hidden=()) -> Session:
        evs = self.events + self.children
        evs.sort(key=lambda e: e.ts_ns)
        return Session(kind, label, evs, set(self.pids), set(hidden))


def _ext_ip(rng: random.Random) -> str:
    return f"{rng.choice([34, 52, 93, 104, 142, 151, 185, 203])}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _shell_activity(g: _Gen, n: int) -> None:
    rng = g.rng
    for _ in range(n):
        cmd = rng.choice(_CMDS)
        if rng.random() < 0.12:  # pipeline / redirect: dup onto stdio is common and benign
            g.emit("dup", arg0=rng.randint(3, 5), arg1=rng.choice([0, 1, 1, 2]))
        g.fork()
        if rng.random() < 0.05:
            g.execve(rng.choice(_SHELLS))
        else:
            g.execve(f"/usr/bin/{cmd}", wait=(0.5, 30))
        if cmd in _NET_CMDS and rng.random() < 0.7:
            g.connect(_ext_ip(rng), rng.choice([443, 443, 80, 22]))
        elif rng.random() < 0.05:
            g.connect("127.0.0.1", rng.choice([5432, 6379, 8080]))
        g.wait(5, 400)


def _benign(kind: str, rng: random.Random, pid: int, t0: int) -> Session:
    g = _Gen(rng, pid, t0)
    if kind == "shell":
        _shell_activity(g, rng.randint(8, 30))
    elif kind == "sudo":
        _shell_activity(g, rng.randint(2, 8))
        for _ in range(rng.randint(1, 3)):
            g.fork()
            g.execve(rng.choice(["/usr/bin/sudo", "/usr/bin/su", "/usr/bin/passwd", "/usr/bin/pkexec"]))
            g.emit("commit_creds", arg0=0, euid=0, wait=(1, 20))
            g.euid = 0
            g.execve(rng.choice(["/usr/bin/apt", "/usr/bin/systemctl", "/usr/bin/vim", "/bin/bash"]))
            g.euid = g.uid
            g.wait(20, 300)
        _shell_activity(g, rng.randint(1, 6))
    elif kind == "root_admin":
        g.uid = g.euid = 0
        _shell_activity(g, rng.randint(8, 25))
    elif kind == "daemon":
        g.uid = g.euid = rng.choice([33, 101, 999])
        g.comm = rng.choice(["nginx", "postgres", "redis-server", "node"])
        for _ in range(rng.randint(15, 45)):
            r = rng.random()
            if r < 0.5:
                g.connect("127.0.0.1", rng.choice([5432, 6379, 11211]))
            elif r < 0.65:
                g.connect(_ext_ip(rng), 443)
            else:
                g.fork()
            g.wait(2, 300)
    elif kind == "build":
        g.comm = "make"
        for _ in range(rng.randint(15, 50)):
            g.fork()
            g.execve(rng.choice(["/usr/bin/gcc", "/usr/bin/ld", "/usr/bin/cc1", "/usr/bin/ar", "/bin/sh"]), wait=(0.2, 8))
            if rng.random() < 0.2:
                g.emit("dup", arg0=4, arg1=rng.choice([1, 2]))
    elif kind == "debugger":
        g.comm = rng.choice(["gdb", "strace", "ltrace"])
        _shell_activity(g, rng.randint(2, 6))
        c = g.fork()
        g.emit("ptrace", arg0=PTRACE_TRACEME, arg1=0)
        g.execve("/usr/bin/" + g.comm)
        foreign = rng.random() < 0.3
        g.emit("ptrace", arg0=PTRACE_ATTACH, arg1=(rng.randint(2000, 9000) if foreign else c))
        for _ in range(rng.randint(2, 8)):
            g.emit("ptrace", arg0=rng.choice([PTRACE_PEEKTEXT, PTRACE_GETREGS, PTRACE_CONT]), arg1=c)
        if not foreign and rng.random() < 0.5:  # gdb breakpoints
            for _ in range(rng.randint(1, 2)):
                g.emit("ptrace", arg0=PTRACE_POKETEXT, arg1=c)
        g.emit("ptrace", arg0=PTRACE_DETACH, arg1=c)
        _shell_activity(g, rng.randint(1, 4))
    elif kind == "preload_dev":
        _shell_activity(g, rng.randint(3, 8))
        g.fork()
        g.execve("/usr/bin/python3", flags=FLAG_LD_PRELOAD_ENV)  # jemalloc / libfaketime style
        _shell_activity(g, rng.randint(2, 8))
    elif kind == "ssh_login":
        g.uid = g.euid = 0
        g.comm = "sshd"
        g.connect("10.0.0.5", 22)
        g.fork()
        g.emit("commit_creds", arg0=1000, euid=1000, wait=(1, 20))  # drop privileges, not escalate
        g.uid = g.euid = 1000
        g.execve(rng.choice(_SHELLS))
        _shell_activity(g, rng.randint(3, 12))
    else:
        raise ValueError(kind)
    return g.finish(kind, BENIGN)


def _attack(kind: str, rng: random.Random, pid: int, t0: int) -> Session:
    g = _Gen(rng, pid, t0)
    _shell_activity(g, rng.randint(2, 12))
    hidden: tuple[int, ...] = ()
    L = kind
    if kind == "privesc":
        if rng.random() < 0.7:  # kernel exploit
            g.execve(rng.choice(["/tmp/.x/exploit", "/dev/shm/pwn", "/tmp/dirty"]), L)
            for _ in range(rng.randint(0, 8)):
                g.fork(L)
            g.emit("commit_creds", L, arg0=0, euid=0, wait=(1, 60))
        else:  # planted setuid binary
            g.fork(L)
            g.execve("/tmp/rootbash", L)
            g.emit("commit_creds", L, arg0=0, euid=0)
        g.euid = 0
        g.execve(rng.choice(_SHELLS), L)
        for _ in range(rng.randint(0, 3)):
            g.execve("/usr/bin/id", L)
    elif kind == "reverse_shell":
        if rng.random() < 0.5:
            g.execve(rng.choice(["/usr/bin/python3", "/usr/bin/perl", "/usr/bin/nc"]), L)
        ip, port = _ext_ip(rng), rng.choice(_ATTACKER_PORTS)
        if rng.random() < 0.85:
            g.emit("socket", L, arg0=2, arg1=1)
        g.emit("connect", L, arg0=3, arg1=port, data=ip, wait=(1, 50))
        for fd in (0, 1, 2) if rng.random() < 0.8 else (0, 1):
            g.emit("dup", L, arg0=3, arg1=fd)
        g.execve(rng.choice(_SHELLS), L)
    elif kind == "ld_preload":
        lib = rng.choice(["/tmp/.libhide.so", "/usr/lib/libsys.so", "/dev/shm/.x.so"])
        if rng.random() < 0.6:
            g.emit("open", L, arg0=0x241, data="/etc/ld.so.preload")
        for _ in range(rng.randint(1, 5)):
            g.fork(L)
            g.execve(rng.choice(["/usr/bin/ls", "/usr/bin/ps", "/usr/sbin/sshd", "/usr/bin/netstat"]), L,
                     flags=FLAG_LD_PRELOAD_ENV if rng.random() < 0.8 else 0)
        if rng.random() < 0.5:
            g.execve(lib, L, flags=FLAG_LD_PRELOAD_ENV)
    elif kind == "ptrace_injection":
        g.execve(rng.choice(["/tmp/injector", "/usr/bin/python3", "/dev/shm/inj"]), L)
        tgt = rng.randint(1200, 9000)
        g.emit("ptrace", L, arg0=rng.choice([PTRACE_ATTACH, PTRACE_ATTACH, PTRACE_SEIZE]), arg1=tgt)
        g.emit("ptrace", L, arg0=PTRACE_GETREGS, arg1=tgt)
        for _ in range(rng.randint(0, 3)):
            g.emit("ptrace", L, arg0=PTRACE_PEEKTEXT, arg1=tgt)
        for _ in range(rng.randint(5, 20)):
            g.emit("ptrace", L, arg0=PTRACE_POKETEXT, arg1=tgt, wait=(0.01, 0.5))
        g.emit("ptrace", L, arg0=PTRACE_SETREGS, arg1=tgt)
        g.emit("ptrace", L, arg0=PTRACE_CONT, arg1=tgt)
        g.emit("ptrace", L, arg0=PTRACE_DETACH, arg1=tgt)
    elif kind == "hidden_pid":
        # A process the kernel reports but /proc does not list (getdents/proc hooking rootkit).
        hid = pid + 5000 + rng.randint(0, 999)
        g.events.clear()
        g.t = t0
        g.pid, g.pids, g.uid, g.euid, g.comm, g.ppid = hid, {hid}, 0, 0, "kworker/u8:3", 2
        ip, port = _ext_ip(rng), rng.choice([443, 8443, 53])
        for _ in range(rng.randint(10, 24)):  # slow C2 beaconing
            g.emit("socket", L, arg0=2, arg1=1, wait=(150, 450))
            g.emit("connect", L, arg0=3, arg1=port, data=ip, wait=(1, 30))
            if rng.random() < 0.3:
                g.execve("/bin/sh", L)
        hidden = (hid,)
        g.pids = {hid}
    else:
        raise ValueError(kind)
    return g.finish(kind, L, hidden)


def make_session(kind: str, rng: random.Random, pid: int, t0: int) -> Session:
    return _attack(kind, rng, pid, t0) if kind in ATTACK_KINDS else _benign(kind, rng, pid, t0)


def random_benign_kind(rng: random.Random) -> str:
    return rng.choices(BENIGN_KINDS, BENIGN_WEIGHTS)[0]


def generate_sessions(n_benign: int, attacks_per_kind: int, seed: int, pid_base: int = 10_000) -> list[Session]:
    """Labelled corpus used for training (benign only) and benchmarking."""
    rng = random.Random(seed)
    out: list[Session] = []
    pid = pid_base
    for _ in range(n_benign):
        out.append(make_session(random_benign_kind(rng), rng, pid, 0))
        pid += 200
    for kind in ATTACK_KINDS:
        for _ in range(attacks_per_kind):
            out.append(make_session(kind, rng, pid, 0))
            pid += 200
    return out
