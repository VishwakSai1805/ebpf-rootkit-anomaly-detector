"""Event schema shared with bpf/events.h (see SPEC.md section 3)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

EVT_NAMES = {
    1: "execve", 2: "ptrace", 3: "commit_creds", 4: "connect", 5: "socket",
    6: "dup", 7: "open", 8: "fork", 9: "exit",
}
EVT_IDS = {v: k for k, v in EVT_NAMES.items()}

FLAG_LD_PRELOAD_ENV = 1 << 0
FLAG_IPV6 = 1 << 1

# ptrace requests (linux/ptrace.h)
PTRACE_TRACEME = 0
PTRACE_PEEKTEXT, PTRACE_PEEKDATA, PTRACE_PEEKUSER = 1, 2, 3
PTRACE_POKETEXT, PTRACE_POKEDATA, PTRACE_POKEUSER = 4, 5, 6
PTRACE_CONT = 7
PTRACE_GETREGS, PTRACE_SETREGS = 12, 13
PTRACE_SETFPREGS = 15
PTRACE_ATTACH, PTRACE_DETACH = 16, 17
PTRACE_SEIZE = 0x4206


@dataclass(slots=True)
class Event:
    ts_ns: int
    pid: int
    type: str
    tid: int = 0
    ppid: int = 0
    uid: int = 0
    euid: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    flags: int = 0
    comm: str = ""
    data: str = ""
    label: str = ""  # ground truth (mock/replay only); never read by the detector

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["label"]:
            del d["label"]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        t = d["type"]
        if isinstance(t, int):
            t = EVT_NAMES.get(t, "unknown")
        return cls(
            ts_ns=int(d["ts_ns"]), pid=int(d["pid"]), type=t,
            tid=int(d.get("tid", d["pid"])), ppid=int(d.get("ppid", 0)),
            uid=int(d.get("uid", 0)), euid=int(d.get("euid", d.get("uid", 0))),
            arg0=int(d.get("arg0", 0)), arg1=int(d.get("arg1", 0)), arg2=int(d.get("arg2", 0)),
            flags=int(d.get("flags", 0)), comm=d.get("comm", ""), data=d.get("data", ""),
            label=d.get("label", ""),
        )

    @classmethod
    def from_json(cls, line: str) -> "Event":
        return cls.from_dict(json.loads(line))
