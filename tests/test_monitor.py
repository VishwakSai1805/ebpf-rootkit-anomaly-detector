import json

import pytest

from cli.monitor import Stats, main, render
from ml.pipeline import Alert


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "model.npz"
    from ml.train import main as train
    assert train(["--out", str(p), "--sessions", "150"]) == 0
    return str(p)


def test_plain_mode_exports_json_alerts(tmp_path, model_path, capsys):
    out = tmp_path / "alerts.jsonl"
    rc = main(["--source", "mock", "--plain", "--rate", "0", "--max-events", "6000", "--attack-rate", "0.5",
               "--seed", "3", "--model", model_path, "--json-out", str(out)])
    assert rc == 0
    lines = out.read_text().splitlines()
    assert lines
    a = json.loads(lines[0])
    assert {"ts_ns", "pid", "score", "categories", "recent"} <= set(a)
    text = capsys.readouterr().out
    assert "alerts in" in text and "[" in text


def test_dashboard_mode_renders(model_path, capsys):
    rc = main(["--source", "mock", "--rate", "0", "--max-events", "3000", "--attack-rate", "0.5", "--no-color",
               "--model", model_path, "--refresh", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "syscall mix" in out and "recent alerts" in out and "inference latency" in out


def test_json_to_stdout_keeps_ui_on_stderr(model_path, capsys):
    main(["--source", "mock", "--plain", "--rate", "0", "--max-events", "4000", "--attack-rate", "0.6",
          "--model", model_path, "--json-out", "-"])
    cap = capsys.readouterr()
    for line in cap.out.splitlines():
        json.loads(line)  # stdout is pure JSONL
    assert "alerts in" in cap.err


def test_record_then_replay(tmp_path, model_path, capsys):
    cap = tmp_path / "c.jsonl"
    assert main(["--source", "mock", "--record", str(cap), "--max-events", "2500", "--attack-rate", "0.5"]) == 0
    assert main(["--source", "replay", "--replay", str(cap), "--plain", "--rate", "0", "--model", model_path]) == 0
    assert "2500 events" in capsys.readouterr().out


def test_replay_missing_file_is_clean_error(capsys):
    assert main(["--source", "replay", "--replay", "/nope.jsonl"]) == 2
    assert "error" in capsys.readouterr().err


def test_render_is_pure_and_handles_empty():
    s = Stats("test")
    frame = render(s, 0, color=False, now=s.t0 + 1)
    assert "none" in frame and "uptime" in frame
    s.events = 10
    s.by_type["execve"] = 10
    s.add_alert(Alert(1, 42, "bash", 1000, 2.5, 0.6, 0.2, ["reverse_shell"], ["connect:1.2.3.4"]))
    frame = render(s, 5, color=True, now=s.t0 + 2)
    assert "reverse_shell" in frame and "critical" in frame
