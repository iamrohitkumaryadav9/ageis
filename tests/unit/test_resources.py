"""Process-tree resource sampler: accounting, identity, and failure handling, on fake trees."""

from __future__ import annotations

import time

import pytest

from aegis.observability.resources import (
    MIB,
    ProcInfo,
    ResourceSampler,
    classify,
    descendants,
    environment_info,
    is_ollama_server,
)

ME = (100, 1000.0)


def proc(pid, ppid, name, rss_mb, *, cmd=None, created=None):
    return ProcInfo(
        pid=pid,
        ppid=ppid,
        name=name,
        cmdline=tuple(cmd if cmd is not None else [name]),
        create_time=created if created is not None else 1000.0 + pid / 1000,
        rss=int(rss_mb * MIB),
    )


def tree(*, runner=True, runner_name="llama-server", extra=()):
    """python(100) -> worker(101); ollama serve(200) -> runner(201); an unrelated process."""
    processes = [
        proc(1, 0, "systemd", 10, created=1.0),
        proc(100, 1, "python", 400, created=1000.0),
        proc(101, 100, "python", 50),
        proc(200, 1, "ollama", 25, cmd=["/opt/ollama/bin/ollama", "serve"]),
        proc(900, 1, "postgres", 300),
    ]
    if runner:
        processes.append(proc(201, 200, runner_name, 3100, cmd=[runner_name, "--model", "x"]))
    return processes + list(extra)


class FakeSource:
    def __init__(self, frames, system=None):
        self.frames = list(frames)
        self.calls = 0
        self._system = system or {
            "used": 2000 * MIB,
            "available": 5000 * MIB,
            "swap_used": 100 * MIB,
            "cpu_percent": 50.0,
        }

    def snapshot(self):
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        if isinstance(frame, Exception):
            raise frame
        return frame

    def system(self):
        if isinstance(self._system, Exception):
            raise self._system
        return dict(self._system)


def sampler(frames, **kwargs):
    return ResourceSampler(
        "test", source=FakeSource(frames, kwargs.pop("system", None)), self_identity=ME, **kwargs
    )


# ----------------------------------------------------------------- classification
def test_runner_is_found_through_the_tree_not_by_its_name():
    # The historical defect: the runner is called "llama-server", so a name match on "ollama"
    # reported 25 MB for a process tree holding 3.1 GB.
    groups = classify(tree(), ME)
    assert groups["ollama_server_rss"] == 25 * MIB
    assert groups["ollama_runner_rss"] == 3100 * MIB
    by_name = sum(p.rss for p in tree() if "ollama" in p.name)
    assert by_name == 25 * MIB  # what the old sampler would have reported


@pytest.mark.parametrize("name", ["llama-server", "ollama", "ollama_llama_server", "runner"])
def test_runner_detection_does_not_depend_on_the_runner_name(name):
    assert classify(tree(runner_name=name), ME)["ollama_runner_rss"] == 3100 * MIB


def test_groups_are_disjoint_and_total_counts_every_process_once():
    groups = classify(tree(), ME)
    assert groups["python_rss"] == 400 * MIB
    assert groups["aegis_tree_rss"] == 450 * MIB  # python + its worker
    assert groups["process_tree_total_rss"] == (450 + 25 + 3100) * MIB
    assert groups["process_tree_total_rss"] == (
        groups["aegis_tree_rss"] + groups["ollama_server_rss"] + groups["ollama_runner_rss"]
    )


def test_unrelated_processes_are_never_counted():
    groups = classify(tree(), ME)
    assert groups["process_tree_total_rss"] < sum(p.rss for p in tree())
    assert 300 * MIB not in groups.values()  # postgres


def test_ollama_started_by_this_process_is_not_double_counted():
    # `ollama serve` is our own child here, so it is in both trees; it must be counted once.
    processes = [
        proc(100, 1, "python", 400, created=1000.0),
        proc(200, 100, "ollama", 25, cmd=["ollama", "serve"]),
        proc(201, 200, "llama-server", 3100),
    ]
    groups = classify(processes, ME)
    assert groups["aegis_tree_rss"] == 400 * MIB
    assert groups["process_tree_total_rss"] == (400 + 25 + 3100) * MIB


def test_ollama_cli_clients_are_not_the_server():
    pull = proc(300, 1, "ollama", 40, cmd=["ollama", "pull", "qwen3:4b"])
    assert not is_ollama_server(pull)
    assert is_ollama_server(proc(200, 1, "ollama", 25, cmd=["/x/ollama", "serve"]))
    assert not is_ollama_server(proc(5, 1, "python", 1, cmd=["python", "serve"]))
    groups = classify(tree(extra=[pull]), ME)
    assert groups["ollama_server_rss"] == 25 * MIB


def test_missing_runner_reports_zero_not_an_error():
    groups = classify(tree(runner=False), ME)
    assert groups["ollama_runner_rss"] == 0 and groups["ollama_runner_processes"] == 0
    assert groups["ollama_server_rss"] == 25 * MIB


def test_no_ollama_at_all():
    groups = classify([proc(100, 1, "python", 400, created=1000.0)], ME)
    assert groups["ollama_server_processes"] == 0
    assert groups["process_tree_total_rss"] == 400 * MIB


def test_nested_descendants_are_followed():
    deep = proc(202, 201, "helper", 7)
    assert classify(tree(extra=[deep]), ME)["ollama_runner_rss"] == 3107 * MIB


def test_reused_pid_does_not_adopt_older_children():
    # PID 200 exited and was reused by a new `ollama serve` created at t=5000. A process that
    # still names 200 as its parent but was created at t=2000 belonged to the old PID 200.
    processes = [
        proc(100, 1, "python", 400, created=1000.0),
        proc(200, 1, "ollama", 25, cmd=["ollama", "serve"], created=5000.0),
        proc(777, 200, "orphan-of-old-pid", 999, created=2000.0),
        proc(201, 200, "llama-server", 3100, created=5001.0),
    ]
    groups = classify(processes, ME)
    assert groups["ollama_runner_rss"] == 3100 * MIB
    assert groups["ollama_runner_processes"] == 1


def test_self_identity_must_match_pid_and_start_time():
    # Same PID, different start time: that is another process, not us.
    impostor = [proc(100, 1, "something-else", 9000, created=4242.0)]
    groups = classify(impostor, ME)
    assert groups["python_rss"] == 0 and groups["self_found"] == 0


def test_descendants_handles_cycles_and_missing_roots():
    looped = [proc(10, 11, "a", 1, created=5.0), proc(11, 10, "b", 1, created=5.0)]
    assert set(descendants({(10, 5.0)}, looped)) == {(11, 5.0)}
    assert descendants({(999, 1.0)}, looped) == {}


# ----------------------------------------------------------------------- sampler
def test_peak_is_taken_per_series_across_samples():
    frames = [
        tree(runner=False),
        tree(),
        tree(extra=[proc(102, 100, "python", 500)]),  # a short-lived worker
        tree(runner=False),
    ]
    s = sampler(frames)
    for _ in frames:
        s.sample_once()
    s._thread, s._started, s._started_wall = _dummy_thread(), s._clock(), _now()
    report = s.stop()
    peaks = report.peaks_mb
    assert peaks["ollama_runner_rss"] == 3100.0
    assert peaks["aegis_tree_rss"] == 950.0
    assert peaks["python_rss"] == 400.0
    assert peaks["process_tree_total_rss"] == 950.0 + 25.0 + 3100.0
    assert peaks["swap_used"] == 100.0
    assert report.process_counts_max == {"ollama_server": 1, "ollama_runner": 1}


def test_terminated_process_between_samples_is_handled():
    s = sampler([tree(), tree(runner=False), [proc(100, 1, "python", 400, created=1000.0)]])
    rows = [s.sample_once() for _ in range(3)]
    assert [r["ollama_runner_rss"] for r in rows] == [3100 * MIB, 0, 0]
    assert rows[2]["ollama_server_rss"] == 0
    assert not s._errors


def test_snapshot_exception_is_counted_and_sampling_continues():
    s = sampler([tree(), RuntimeError("procfs exploded"), tree()])
    assert s.sample_once() is not None
    assert s.sample_once() is None
    assert s.sample_once() is not None
    assert s._errors == {"process_snapshot:RuntimeError": 1}
    assert len(s._rows) == 2


def test_system_metric_failure_keeps_process_data_and_is_reported():
    s = sampler([tree()], system=OSError("no /proc/meminfo"))
    row = s.sample_once()
    assert row["ollama_runner_rss"] == 3100 * MIB
    s._thread, s._started, s._started_wall = _dummy_thread(), s._clock(), _now()
    report = s.stop()
    assert report.peaks_mb["system_used"] is None  # unavailable, not zero
    assert report.cpu_percent["mean"] is None
    assert report.errors["system_metrics:OSError"] >= 1
    assert report.to_dict()["monitoring_error_total"] >= 1


def test_losing_track_of_our_own_process_is_an_error():
    s = sampler([[proc(5, 1, "other", 1)]])
    s.sample_once()
    assert s._errors["self_process_not_found"] == 1


def test_background_thread_samples_at_the_configured_interval():
    s = sampler([tree()], interval_s=0.05)
    with s:
        time.sleep(0.6)
    report = s.stop()
    assert report.samples >= 8  # t=0, ~10 ticks, final
    assert report.sampling_interval_s == 0.05
    assert report.duration_s >= 0.5
    assert report.errors == {}
    assert report.run_id.startswith("RES-") and report.label == "test"
    assert report.started_at <= report.ended_at


def test_stop_is_idempotent_and_start_is_single_use():
    s = sampler([tree()], interval_s=0.05)
    s.start()
    first = s.stop()
    assert s.stop() is first
    with pytest.raises(RuntimeError, match="already started"):
        s.start()


def test_stop_without_start_is_refused():
    with pytest.raises(RuntimeError, match="never started"):
        sampler([tree()]).stop()


@pytest.mark.parametrize("interval", [0, -1, 0.001, float("nan")])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="interval"):
        ResourceSampler("x", interval_s=interval, source=FakeSource([tree()]), self_identity=ME)


@pytest.mark.parametrize("label", ["", "   "])
def test_label_is_required(label):
    with pytest.raises(ValueError, match="label"):
        ResourceSampler(label, source=FakeSource([tree()]), self_identity=ME)


def test_run_without_ollama_says_so_in_the_report():
    s = sampler([[proc(100, 1, "python", 400, created=1000.0)]], interval_s=0.05)
    with s:
        pass
    notes = " ".join(s.stop().notes)
    assert "no Ollama runner" in notes and "no Ollama server" in notes


def test_two_samplers_have_distinct_run_ids():
    assert sampler([tree()]).run_id != sampler([tree()]).run_id


def test_report_is_json_serialisable():
    import json

    s = sampler([tree()], interval_s=0.05)
    with s:
        pass
    payload = json.loads(json.dumps(s.stop().to_dict()))
    assert set(payload["peaks_mb"]) == {
        "python_rss",
        "aegis_tree_rss",
        "ollama_server_rss",
        "ollama_runner_rss",
        "process_tree_total_rss",
        "system_used",
        "swap_used",
    }


# ------------------------------------------------------------------- environment
def test_environment_reports_memory_in_both_unit_systems():
    info = environment_info(include_host=False)
    total = info["guest_memory_total_bytes"]
    assert info["guest_memory_total_gib"] == round(total / 2**30, 2)
    assert info["guest_memory_total_gb_decimal"] == round(total / 1e9, 2)
    assert info["guest_memory_total_gb_decimal"] > info["guest_memory_total_gib"]
    assert info["logical_cpus"] >= info["physical_cores"] >= 1


def test_host_facts_never_invent_a_value(monkeypatch):
    from aegis.observability import resources

    monkeypatch.setattr(resources, "is_wsl", lambda: True)
    monkeypatch.setattr(resources, "_powershell", lambda *a, **k: None)
    facts = resources.host_facts()
    assert facts["physical_ram_bytes"] is None
    assert "unavailable" in facts["physical_ram_source"]
    assert facts["wslconfig"] is None


def test_host_facts_parse_the_host_answer(monkeypatch):
    from aegis.observability import resources

    answers = iter(["16957034496", "ABSENT"])
    monkeypatch.setattr(resources, "is_wsl", lambda: True)
    monkeypatch.setattr(resources, "_powershell", lambda *a, **k: next(answers))
    facts = resources.host_facts()
    assert facts["physical_ram_bytes"] == 16957034496
    assert facts["wslconfig"].startswith("absent")


def test_host_facts_reject_a_non_numeric_answer(monkeypatch):
    from aegis.observability import resources

    monkeypatch.setattr(resources, "is_wsl", lambda: True)
    monkeypatch.setattr(resources, "_powershell", lambda *a, **k: "Access denied")
    assert resources.host_facts()["physical_ram_bytes"] is None


def _dummy_thread():
    import threading

    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    return thread


def _now():
    import datetime as dt

    return dt.datetime.now(dt.UTC)
