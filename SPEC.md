# SPEC — Kernel-to-Userspace Architecture

## 1. Goal
Observe security-relevant syscalls/kernel functions with eBPF, turn them into behavioural
features, and flag rootkit-like / post-exploitation behaviour with an *unsupervised* model
trained only on benign activity, at <1 ms inference latency per window.

## 2. Data flow
```
 kernel                                   userspace
 ┌──────────────────────────────┐   ringbuf   ┌───────────────┐  Event  ┌──────────────────────┐  windows  ┌───────────────┐
 │ tracepoints + kprobes        │ ──────────▶ │ bpf/loader.c  │ ──────▶ │ collector/sources.py │ ────────▶ │ ml/pipeline.py│
 │  sys_enter_execve/ptrace/... │  (struct    │ (libbpf, JSONL│  JSONL  │ Live | Replay | Mock │           │ features →    │
 │  kprobe commit_creds         │   event)    │  on stdout)   │         └──────────────────────┘           │ IF + AE → Alert│
 │  sched fork/exit             │             └───────────────┘                  ▲ /proc snapshot          └──────┬────────┘
 └──────────────────────────────┘                                                │                               ▼
                                                                         hidden-PID cross-check        cli/monitor.py (TUI + JSONL)
```

## 3. Event ABI (`bpf/events.h` ⇄ `collector/events.py`)
| field | type | meaning |
|---|---|---|
| `ts_ns` | u64 | `bpf_ktime_get_ns()` |
| `pid`,`tid`,`ppid` | u32 | tgid, tid, real parent tgid |
| `uid`,`euid` | u32 | task real uid / effective uid (`task->cred`); for `commit_creds`, `euid` is the **new** cred euid |
| `type` | u32 | 1 execve, 2 ptrace, 3 commit_creds, 4 connect, 5 socket, 6 dup, 7 open, 8 fork, 9 exit |
| `arg0..2` | s64 | execve: –; ptrace: request, target pid; connect: fd, port; dup: oldfd, newfd; fork: child pid; socket: domain, type |
| `flags` | u32 | bit0 `LD_PRELOAD` found in execve envp; bit1 IPv6 connect |
| `comm` | char[16] | task name |
| `data` | char[96] | execve filename / open path / connect IPv4 text (userspace form) |

Loader output is one JSON object per line using the same names (`type` as string).

## 4. Probes (`bpf/probes.bpf.c`)
`tp/syscalls/sys_enter_execve` (filename + bounded envp scan for `LD_PRELOAD=`),
`sys_enter_ptrace`, `kprobe/commit_creds`, `sys_enter_connect`, `sys_enter_socket`,
`sys_enter_dup2/dup3`, `sys_enter_openat` (kernel-side filter: `/etc/ld.so*`),
`sched_process_fork`, `sched_process_exit`. CO-RE (`preserve_access_index`) for `task_struct`/`cred`; no `vmlinux.h` needed.
Events leave via one `BPF_MAP_TYPE_RINGBUF`; drops are counted in a per-CPU map.

## 5. Collector modes
`LiveSource` (spawns the loader; needs CAP_BPF/CAP_SYS_ADMIN) → `ReplaySource` (JSONL file) →
`MockSource` (deterministic synthetic traffic with injected attacks). `auto_source()` picks Live
if the loader + capabilities exist, otherwise Mock, so CI/containers always work.

## 6. Features (`ml/features.py`)
Per-PID sliding window (default 32 events, stride 1 online).
- Event tokens (16-symbol alphabet) → unigram (16), bigram (256), hashed trigram (64) frequencies.
- Scalars: UID→root transition, *unexplained* root transition (no setuid exec just before), exec from `/tmp` or `/dev/shm`,
  ptrace attach / poke count / setregs / foreign target, `LD_PRELOAD` env + `ld.so.preload` open,
  external connects, stdio `dup`s, reverse-shell chain (`connect_ext → dup→stdio → shell exec`), hidden PIDs.
- **Hidden-PID cross-check**: PIDs the kernel reports (events, fork child pids) minus PIDs listed in `/proc`
  (minus PIDs that exited). Non-empty ⇒ process hiding (e.g. `getdents` hooking).

## 7. Model (`ml/detector.py`)
- Isolation Forest (sklearn for training) exported to flat arrays; inference = vectorised numpy tree walk.
- Autoencoder (numpy, Adam, D→64→16→64→D), reconstruction MSE on clipped z-scores.
- Both trained on benign windows only; thresholds = benign-validation percentile. Alert if either exceeds its threshold;
  fused score = max(score/threshold). Explanations come from feature flags → categories
  (`privilege_escalation`, `reverse_shell`, `ld_preload_injection`, `ptrace_injection`, `hidden_process`).
- Persisted as `.npz` (no pickle).

## 8. Outputs
`cli/monitor.py`: ANSI dashboard or `--plain` log, JSONL alerts (`--json-out`), event recording (`--record`) for replay.
`benchmarks/run_benchmark.py`: precision/recall/F1, per-class recall, latency percentiles → `reports/benchmark_results.md`.

## 9. Non-goals / honesty
Mock traffic is synthetic; benchmark numbers describe separability on that generator, not production
false-positive rates. Live mode has been designed for kernels ≥5.8 with BTF but is only compile-checked in CI.
