import subprocess
import sys


def test_pipeline_runs_without_an_api_key_present(tmp_path, monkeypatch):
    """The demo must survive a missing key or a dead network."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = subprocess.run(
        [sys.executable, "reconcile.py", "--data", "data",
         "--out", str(tmp_path), "--no-llm"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "predicted_matches.csv").exists()


def test_explain_prints_a_decision_trace(tmp_path):
    result = subprocess.run(
        [sys.executable, "reconcile.py", "--data", "data", "--out", str(tmp_path),
         "--no-llm", "--explain", "TXN-000001"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "match_id" in result.stdout
