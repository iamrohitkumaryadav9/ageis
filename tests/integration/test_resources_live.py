"""The sampler against the real process table (no fakes)."""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest

from aegis.config import get_settings
from aegis.observability.resources import ResourceSampler, environment_info

CHILD = "import time; x = bytearray(120 * 1024 * 1024); print('ready', flush=True); time.sleep(30)"


def test_a_real_child_process_is_counted_in_the_tree_and_released_when_it_exits():
    sampler = ResourceSampler("live-child", interval_s=0.1).start()
    child = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE)
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == b"ready"
        time.sleep(0.5)
        during = sampler.sample_once()
    finally:
        child.kill()
        child.wait(timeout=10)
    time.sleep(0.3)
    after = sampler.sample_once()
    report = sampler.stop()
    assert during is not None and after is not None
    child_mb = (during["aegis_tree_rss"] - during["python_rss"]) / 2**20
    assert child_mb >= 110, f"child holding 120 MiB was measured at {child_mb:.0f} MiB"
    assert (after["aegis_tree_rss"] - after["python_rss"]) / 2**20 < 20  # gone after exit
    assert report.peaks_mb["aegis_tree_rss"] >= report.peaks_mb["python_rss"] + 110
    assert report.peaks_mb["process_tree_total_rss"] >= report.peaks_mb["aegis_tree_rss"]
    assert report.errors == {}
    assert report.peaks_mb["system_used"] > 0 and report.cpu_percent["mean"] is not None


def test_environment_info_matches_the_operating_system():
    info = environment_info(include_host=False)
    with open("/proc/meminfo") as handle:
        kb = int(next(line for line in handle if line.startswith("MemTotal")).split()[1])
    assert info["guest_memory_total_bytes"] == kb * 1024


@pytest.mark.ollama
def test_loaded_model_is_attributed_to_the_runner_not_the_server():
    """Needs a running Ollama with the configured model. Loads it, then measures."""
    settings = get_settings()
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "keep_alive": "1m"},
        timeout=settings.ollama_load_timeout_s,
    )
    assert response.status_code == 200, response.text[:200]
    sampler = ResourceSampler("live-ollama", interval_s=0.2)
    with sampler:
        time.sleep(1.0)
    report = sampler.stop()
    assert report.process_counts_max["ollama_server"] >= 1
    assert report.process_counts_max["ollama_runner"] >= 1
    assert report.peaks_mb["ollama_runner_rss"] > 1000, report.peaks_mb
    assert report.peaks_mb["ollama_server_rss"] < 500, report.peaks_mb
    assert report.errors == {}
