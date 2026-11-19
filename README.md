# eBPF Syscall Telemetry & Unsupervised Rootkit/Anomaly Detection Engine

Kernel-level telemetry (eBPF) → behavioural features → unsupervised ML (Isolation Forest + Autoencoder) → real-time
alerts. Detects **privilege escalation, reverse shells, `LD_PRELOAD` / `ptrace` injection and hidden processes** with
sub-millisecond inference, and runs end-to-end **without kernel privileges** thanks to a deterministic mock/replay mode.

```
 kernel (eBPF)                      userspace
 tracepoints/kprobes ─ringbuf─▶ loader ─JSONL─▶ collector ─▶ per-PID windows ─▶ features ─▶ IF + AE ─▶ alerts
 execve ptrace commit_creds         (libbpf)     Live | Replay | Mock    ▲ /proc snapshot             │
 connect socket dup openat fork/exit                                      └ hidden-PID cross-check     ▼
                                                                                          dashboard + JSONL
```

## Highlights
- **Probes** (`bpf/probes.bpf.c`): `sys_enter_execve` (+ in-kernel `LD_PRELOAD=` envp scan), `sys_enter_ptrace`, `kprobe/commit_creds`,
  `sys_enter_connect/socket/dup2/dup3`, filtered `openat` (`/etc/ld.so*`), `sched_process_fork/exit`. CO-RE without `vmlinux.h`.
  The object compiles with clang and passes the in-kernel verifier (checked in the dev container; attaching needs tracefs + CAP_BPF/CAP_SYS_ADMIN).
- **Collector** (`collector/`): `LiveSource` (libbpf loader → JSONL), `ReplaySource`, `MockSource`; `auto_source` falls back to mock when kernel
  capabilities are missing, so tests and demos run anywhere (CI, containers).
- **Features** (`ml/features.py`): per-PID 32-event sliding windows; unigram/bigram/hashed-trigram frequencies over a 20-symbol syscall alphabet;
  UID→root transition flags (and whether a setuid exec *explains* them); exec-from-`/tmp`; ptrace attach/poke/setregs/foreign-target;
  `LD_PRELOAD` env + `ld.so.preload` opens; reverse-shell chain (connected socket fd duplicated onto stdio, then a shell exec);
  **hidden-PID cross-check** (kernel-seen PIDs vs `/proc`, with exit tracking and a race grace period).
- **Detector** (`ml/detector.py`): Isolation Forest (sklearn-trained, exported to flat arrays) + numpy autoencoder, trained on benign data only;
  thresholds from held-out benign percentiles; per-alert category attribution; `.npz` persistence (no pickle). Inference is pure numpy (~130 µs p50).
- **CLI** (`cli/monitor.py`): ANSI dashboard or plain log, JSONL alert export, record/replay.
- **Benchmarks** (`benchmarks/`): precision/recall/F1, per-class recall, time-to-detect, ablations, latency → [`reports/benchmark_results.md`](reports/benchmark_results.md).

## Results (synthetic traffic, 3 held-out corpora; full table in the report)
| | precision | recall | F1 |
|---|---|---|---|
| session level, fused | 0.934 | 0.937 | 0.935 |
| window level, fused | 0.980 | 0.909 | 0.943 |

Per-class session recall: privesc 1.00, ptrace injection 1.00, hidden PID 1.00, reverse shell 0.93, `LD_PRELOAD` 0.76 (one attack variant is
indistinguishable from a benign dev-tool pattern, by construction). Inference p99 ≈ 250 µs; end-to-end (features + inference) p99 ≈ 0.7 ms.

> **Honesty note.** Traffic is synthetic. The benign generator includes hard negatives (sudo, gdb/strace, `LD_PRELOAD` tooling, sshd privilege drops, pipelines),
> but attack chains are idealised, and the ablation shows the autoencoder does most of the work (the forest adds false positives without adding recall
> on this data). Treat the numbers as evidence that the pipeline works and separates these behaviours, not as production FP rates.
> Benign-session FPR (any window flagged) is ~8%; per-window FPR is ~0.7%.

## Quick start
```bash
sudo apt-get install -y clang llvm libbpf-dev gcc make python3-pytest python3-sklearn python3-numpy   # or: pip install numpy scikit-learn pytest
pytest -q --tb=short                                   # 50 tests, no privileges needed
python -m cli.monitor --source mock --duration 10      # live dashboard on synthetic traffic with injected attacks
python -m cli.monitor --source mock --plain --max-events 20000 --json-out alerts.jsonl
python -m benchmarks.run_benchmark                     # regenerate reports/benchmark_results.md
```
### Real kernel telemetry (privileged host, kernel ≥ 5.8 with BTF)
```bash
make bpf                                               # -> bpf/build/probes.bpf.o and bpf/build/loader
sudo python -m cli.monitor --source live --json-out /var/log/ebpf-alerts.jsonl
sudo python -m cli.monitor --source live --record capture.jsonl --max-events 50000   # capture for offline replay
python -m cli.monitor --source replay --replay capture.jsonl --plain
```
`--source auto` (default) tries live and silently falls back to mock. Retrain on your own benign baseline with `python -m ml.train --out build/model.npz`
(on live captures, build a matrix with `ml.dataset.build_matrix`-style code from replayed events).

## Alert format (JSONL)
```json
{"ts_ns":1042113000,"pid":23800,"comm":"bash","uid":1000,"score":39.2,"if_score":0.55,"ae_error":3.4,
 "categories":["privilege_escalation"],"recent":["fork:23811","commit_creds:0"],"truth":"privesc"}
```
`score ≥ 1` is anomalous (max of per-model score/threshold); severity: ≥2 critical, ≥1.3 high, else medium. `truth` is only populated for mock/replay data.

## Repository layout
```
bpf/        events.h probes.bpf.c loader.c        collector/  events.py sources.py scenarios.py
ml/         features.py detector.py dataset.py pipeline.py train.py
cli/        monitor.py     benchmarks/ run_benchmark.py     reports/ benchmark_results.{md,json}
tests/      collector, anomaly pipeline, monitor, benchmark     SPEC.md  CLAUDE.md
```
See [`SPEC.md`](SPEC.md) for the event ABI and design details.

## Design decisions & limitations
- **Why custom numpy inference?** `sklearn.predict` costs milliseconds per call; exporting trees to arrays gives ~100 µs and keeps the hot path dependency-light.
- **Scalar up-weighting in the autoencoder** keeps rare, high-signal flags (e.g. ptrace pokes) from being drowned by ~480 sparse n-gram dimensions.
- **Simplifications:** synthetic fork+exec is attributed to the parent PID; `commit_creds` fires for every credential commit (setuid exec included), hence the "explained by a setuid exec" feature;
  IPv6 connects are flagged but not classified; the live path was compile- and verifier-checked here but could not attach in the unprivileged container (no tracefs).
- **Evasion:** an attacker with kernel access can tamper with eBPF itself; unsupervised models can be poisoned by a malicious baseline; slow-and-low behaviour spread over windows is not modelled.

## License
MIT
