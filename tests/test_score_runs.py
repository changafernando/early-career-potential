"""Smoke test: `run_pipeline.py score` runs end to end on the registered models, writing only to a temp copy."""
import json
import shutil
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT


def test_score_command_runs_on_the_registered_models(tmp_path):
    if not (ROOT / "outputs" / "metrics.json").exists():
        pytest.skip("run the pipeline first")
    shutil.copytree(ROOT / "outputs", tmp_path / "outputs")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["paths"] = {**cfg["paths"], "processed_dir": str(ROOT / "data" / "processed"),
                    "models_dir": str(ROOT / "models"), "outputs_dir": str(tmp_path / "outputs"),
                    "dashboard": str(tmp_path / "ecp_dashboard.html")}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    run = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "score", "--config", str(tmp_path / "config.yaml")],
                         capture_output=True, text=True, timeout=300)

    assert run.returncode == 0, run.stderr[-2000:]
    assert "scored" in run.stdout
    live = json.loads((tmp_path / "outputs" / "watchlist_current.json").read_text())
    assert live and all(0 <= r["p"] <= 1 for r in live)
    assert (tmp_path / "ecp_dashboard.html").stat().st_size > 0
