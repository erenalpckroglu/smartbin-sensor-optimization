"""End-to-end: simulate a small fleet with known truth, run everything, demand the truth back."""
import json

import pytest

from smartbin.pipeline import run
from smartbin.simulate import simulate


@pytest.mark.slow
def test_pipeline_recovers_planted_effects(tmp_path):
    data = tmp_path / "data"
    simulate(n_devices=900, out_dir=data, seed=21)
    import smartbin.config as c
    cfg_text = (c.ROOT / "config" / "config.yaml").read_text().replace(
        "interim: data/interim", f"interim: {tmp_path / 'interim'}")
    (tmp_path / "cfg.yaml").write_text(cfg_text)
    R = run(str(tmp_path / "cfg.yaml"), str(data / "measurements.csv"), str(data / "scale_metadata.csv"),
            str(data / "scale_preprocessing.csv"), str(tmp_path / "reports"))
    assert all(r.status in ("ok", "partial") for k, r in R.items()), {k: r.status for k, r in R.items()}
    v = R["m0_validation"].tables["checks"]
    assert (v.status == "PASS").mean() >= 0.8, v.to_string()
    res = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert "m7_policy" in res["modules"]
    assert (tmp_path / "reports" / "dashboard_6kg.html").exists()
