# CLAUDE.md — working notes for Claude Code

Project: **eBPF Syscall Telemetry & Unsupervised ML Rootkit/Anomaly Detection Engine**.
Read `SPEC.md` first; it is the source of truth for the event schema and architecture.

## Layout
- `bpf/` — kernel side. `events.h` (shared ABI), `probes.bpf.c` (CO-RE programs), `loader.c` (libbpf loader that prints JSONL).
- `collector/` — userspace ingestion: `events.py` (Event schema), `sources.py` (Live / Replay / Mock + `auto_source`), `scenarios.py` (labelled synthetic benign + attack traffic).
- `ml/` — `features.py` (windowed n-grams, UID flags, hidden-PID cross-check), `detector.py` (Isolation Forest + numpy Autoencoder), `dataset.py`, `pipeline.py` (streaming engine, alerts), `train.py`.
- `cli/monitor.py` — live dashboard + JSON alert exporter.
- `benchmarks/run_benchmark.py` → `reports/benchmark_results.md`.
- `tests/` — pytest suite. Must pass with **no** kernel privileges (mock/replay only).

## Commands
```bash
make bpf            # build bpf/build/probes.bpf.o + loader (needs clang, libbpf-dev; optional)
pytest -q --tb=short
python -m ml.train --out build/model.npz
python -m cli.monitor --source mock --plain --duration 5
python -m benchmarks.run_benchmark
```

## Rules
- Tests must never require root/CAP_BPF. Live mode is auto-detected; `auto_source` falls back to mock.
- The Python `Event` schema and `bpf/events.h` must change together (see SPEC.md §3).
- Detector inference (features -> score) must stay < 1 ms median; `tests/test_anomaly_pipeline.py` enforces it. Keep inference numpy-only (no sklearn `predict` calls on the hot path).
- Everything is deterministic given a seed; keep it that way (benchmarks are committed).
- Build artifacts (`build/`, `*.o`, `*.so`, `*.onnx`, `venv/`, caches) are git-ignored; never commit them.
- Commit per milestone; keep commits focused.
