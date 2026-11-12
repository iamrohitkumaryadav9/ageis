"""Process-tree resource measurement for benchmark and verification runs.

What is measured, and how the pieces relate:

* ``python``        resident memory of this process alone
* ``aegis_tree``    this process plus every descendant it spawned
* ``ollama_server`` every process running ``ollama serve`` (it holds no model)
* ``ollama_runner`` the server's descendants: the runner that maps the model into memory. Ollama
                    0.40 names it ``llama-server``, so it must be found through the process
                    tree, never by name
* ``process_tree_total`` = ``aegis_tree`` + ``ollama_server`` + ``ollama_runner``, with every
                    process counted exactly once, keyed by (pid, create time)
* ``system_used`` / ``swap_used``  machine-wide, independent of the above

Resident set sizes are summed, so pages shared between processes (mostly shared libraries) are
counted in each process that maps them. The total is therefore an upper bound on what the tree
occupies; ``system_used`` is the cross-check.

The sampler never raises into the measured workload. Every failure is counted in ``errors`` and
reported, so a run whose sampler misbehaved cannot pass for a clean measurement.

Processes are read through a small ``ProcessSource`` interface so that tests can supply fake
process trees: vanished processes, missing children, reused PIDs.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil

from aegis.observability.logging import get_logger

log = get_logger(__name__)

MIB = 1024 * 1024
DEFAULT_INTERVAL_S = 0.5
MIN_INTERVAL_S = 0.05
SERIES = (
    "python_rss",
    "aegis_tree_rss",
    "ollama_server_rss",
    "ollama_runner_rss",
    "process_tree_total_rss",
    "system_used",
    "system_available",
    "swap_used",
    "cpu_percent",
)


class ProcessGone(Exception):
    """The process exited (or cannot be read) between being listed and being inspected."""


@dataclass(frozen=True)
class ProcInfo:
    """One process at one instant. (pid, create_time) identifies it across samples."""

    pid: int
    ppid: int
    name: str
    cmdline: tuple[str, ...]
    create_time: float
    rss: int

    @property
    def identity(self) -> tuple[int, float]:
        return (self.pid, round(self.create_time, 2))


class ProcessSource(Protocol):
    def snapshot(self) -> list[ProcInfo]:
        """Every readable process right now. Processes that vanish mid-listing are skipped."""
        ...

    def system(self) -> dict[str, float]:
        """Machine-wide ``used``, ``available``, ``swap_used`` in bytes and ``cpu_percent``."""
        ...


class PsutilSource:
    """The real process table."""

    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)  # prime: the first call always returns 0.0

    def snapshot(self) -> list[ProcInfo]:
        processes = []
        for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
            try:
                info = proc.info
                processes.append(
                    ProcInfo(
                        pid=info["pid"],
                        ppid=info["ppid"] or 0,
                        name=info["name"] or "",
                        cmdline=tuple(info["cmdline"] or ()),
                        create_time=info["create_time"] or 0.0,
                        rss=proc.memory_info().rss,
                    )
                )
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue  # exited or unreadable between listing and reading: not an error
        return processes

    def system(self) -> dict[str, float]:
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        return {
            "used": float(memory.used),
            "available": float(memory.available),
            "swap_used": float(swap.used),
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
        }


def is_ollama_server(proc: ProcInfo) -> bool:
    """``ollama serve``; not the ``ollama pull`` / ``ollama run`` command-line clients."""
    executable = os.path.basename(proc.cmdline[0]) if proc.cmdline else proc.name
    return executable.lower().startswith("ollama") and "serve" in proc.cmdline[1:]


def descendants(
    roots: set[tuple[int, float]], processes: list[ProcInfo]
) -> dict[tuple[int, float], ProcInfo]:
    """Every process below ``roots`` in the parent/child tree, excluding the roots themselves.

    A child must have been created no earlier than its parent. That one comparison is what keeps
    a recycled PID from adopting somebody else's children.
    """
    by_pid = {p.pid: p for p in processes}
    children: dict[int, list[ProcInfo]] = {}
    for proc in processes:
        children.setdefault(proc.ppid, []).append(proc)
    found: dict[tuple[int, float], ProcInfo] = {}
    frontier = [by_pid[pid] for pid, _ in roots if pid in by_pid]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent.pid, []):
            if child.identity in found or child.identity in roots:
                continue
            if child.create_time + 1.0 < parent.create_time:
                continue  # older than its "parent": the parent's PID was reused
            found[child.identity] = child
            frontier.append(child)
    return found


def classify(processes: list[ProcInfo], self_identity: tuple[int, float]) -> dict[str, int]:
    """Split one snapshot into the reported groups. No process is counted in two groups."""
    by_identity = {p.identity: p for p in processes}
    me = by_identity.get(self_identity)
    servers = {p.identity: p for p in processes if is_ollama_server(p)}
    runners = descendants(set(servers), processes)
    aegis = descendants({self_identity}, processes) if me else {}
    # If Ollama was started by this process it is reported as Ollama, not twice.
    aegis = {k: v for k, v in aegis.items() if k not in servers and k not in runners}
    python_rss = me.rss if me else 0
    aegis_tree = python_rss + sum(p.rss for p in aegis.values())
    server_rss = sum(p.rss for k, p in servers.items() if k != self_identity)
    runner_rss = sum(p.rss for k, p in runners.items() if k != self_identity)
    return {
        "python_rss": python_rss,
        "aegis_tree_rss": aegis_tree,
        "ollama_server_rss": server_rss,
        "ollama_runner_rss": runner_rss,
        "process_tree_total_rss": aegis_tree + server_rss + runner_rss,
        "ollama_server_processes": len(servers),
        "ollama_runner_processes": len(runners),
        "self_found": int(me is not None),
    }


@dataclass
class ResourceReport:
    """Everything one measured run produced. ``to_dict`` is what gets written to disk."""

    run_id: str
    label: str
    started_at: str
    ended_at: str
    duration_s: float
    sampling_interval_s: float
    samples: int
    peaks_mb: dict[str, float | None]
    at_start_mb: dict[str, float | None]
    cpu_percent: dict[str, float | None]
    process_counts_max: dict[str, int]
    errors: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "sampling_interval_s": self.sampling_interval_s,
            "samples": self.samples,
            "peaks_mb": self.peaks_mb,
            "at_start_mb": self.at_start_mb,
            "cpu_percent": self.cpu_percent,
            "process_counts_max": self.process_counts_max,
            "monitoring_errors": self.errors,
            "monitoring_error_total": sum(self.errors.values()),
            "notes": self.notes,
        }


class ResourceSampler:
    """Samples the process tree on a background thread until stopped.

    ``label`` names what is being measured (for example ``deterministic`` or ``llm``); it is
    stored with the result so two runs can never be confused after the fact.
    """

    def __init__(
        self,
        label: str,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        source: ProcessSource | None = None,
        self_identity: tuple[int, float] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not label or not label.strip():
            raise ValueError("a resource measurement needs a non-empty label")
        if not interval_s >= MIN_INTERVAL_S:
            raise ValueError(f"sampling interval must be at least {MIN_INTERVAL_S} s")
        self.label = label.strip()
        self.interval_s = float(interval_s)
        self.run_id = f"RES-{uuid.uuid4().hex[:12]}"
        self._source = source or PsutilSource()
        if self_identity is None:
            me = psutil.Process()
            self_identity = (me.pid, round(me.create_time(), 2))
        self._self = self_identity
        self._clock = clock
        self._rows: list[dict[str, float]] = []
        self._errors: Counter[str] = Counter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_wall: dt.datetime | None = None
        self._started: float | None = None
        self._report: ResourceReport | None = None

    # ------------------------------------------------------------------ sampling
    def sample_once(self) -> dict[str, float] | None:
        """Take one sample. Returns ``None`` (and counts an error) if it could not be taken."""
        try:
            row: dict[str, float] = {
                k: float(v) for k, v in classify(self._source.snapshot(), self._self).items()
            }
        except Exception as exc:  # the sampler must never take the workload down with it
            self._errors[f"process_snapshot:{type(exc).__name__}"] += 1
            return None
        if not row.pop("self_found"):
            self._errors["self_process_not_found"] += 1
        try:
            system = self._source.system()
            row.update(
                system_used=system["used"],
                system_available=system["available"],
                swap_used=system["swap_used"],
                cpu_percent=system["cpu_percent"],
            )
        except Exception as exc:
            self._errors[f"system_metrics:{type(exc).__name__}"] += 1
            row.update(
                system_used=np.nan, system_available=np.nan, swap_used=np.nan, cpu_percent=np.nan
            )
        self._rows.append(row)
        return row

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample_once()

    def start(self) -> ResourceSampler:
        if self._thread is not None:
            raise RuntimeError("this sampler was already started; create a new one per run")
        self._started_wall = dt.datetime.now(dt.UTC)
        self._started = self._clock()
        self.sample_once()  # a sample at t=0, so even a very short run has a measurement
        self._thread = threading.Thread(
            target=self._loop, name=f"resource-sampler-{self.label}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> ResourceReport:
        if self._report is not None:
            return self._report  # idempotent: a second stop returns the same report
        if self._thread is None or self._started is None or self._started_wall is None:
            raise RuntimeError("the sampler was never started")
        self._stop.set()
        self._thread.join(timeout=max(5.0, 4 * self.interval_s))
        if self._thread.is_alive():
            self._errors["sampler_thread_did_not_stop"] += 1
        self.sample_once()  # and one at the very end
        duration = self._clock() - self._started
        self._report = self._summarise(duration)
        if self._report.errors:
            log.warning(
                "resource_sampler_errors", run_id=self.run_id, errors=dict(self._report.errors)
            )
        return self._report

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ summary
    def _summarise(self, duration: float) -> ResourceReport:
        rows = self._rows

        def column(key: str) -> np.ndarray:
            return np.array([r.get(key, np.nan) for r in rows], dtype=float)

        def peak_mb(key: str) -> float | None:
            values = column(key)
            values = values[~np.isnan(values)]
            return round(float(values.max()) / MIB, 1) if values.size else None

        def first_mb(key: str) -> float | None:
            values = column(key)
            values = values[~np.isnan(values)]
            return round(float(values[0]) / MIB, 1) if values.size else None

        cpu = column("cpu_percent")
        cpu = cpu[~np.isnan(cpu)][1:]  # the first reading after priming covers no interval
        notes = []
        if rows and not any(r.get("ollama_runner_processes") for r in rows):
            notes.append("no Ollama runner process was observed during this run")
        if rows and not any(r.get("ollama_server_processes") for r in rows):
            notes.append("no Ollama server process was observed during this run")
        expected = duration / self.interval_s if self.interval_s else 0
        if expected >= 10 and len(rows) < 0.5 * expected:
            self._errors["fewer_samples_than_expected"] += 1
            notes.append(
                f"{len(rows)} samples taken where about {expected:.0f} were expected: "
                "short peaks may have been missed"
            )
        assert self._started_wall is not None
        return ResourceReport(
            run_id=self.run_id,
            label=self.label,
            started_at=self._started_wall.isoformat(timespec="seconds"),
            ended_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            duration_s=round(duration, 1),
            sampling_interval_s=self.interval_s,
            samples=len(rows),
            peaks_mb={
                key: peak_mb(key)
                for key in SERIES
                if key not in ("cpu_percent", "system_available")
            },
            at_start_mb={
                "system_available": first_mb("system_available"),
                "system_used": first_mb("system_used"),
                "swap_used": first_mb("swap_used"),
            },
            cpu_percent={
                "mean": round(float(cpu.mean()), 1) if cpu.size else None,
                "p95": round(float(np.percentile(cpu, 95)), 1) if cpu.size else None,
                "max": round(float(cpu.max()), 1) if cpu.size else None,
            },
            process_counts_max={
                "ollama_server": int(max((r["ollama_server_processes"] for r in rows), default=0)),
                "ollama_runner": int(max((r["ollama_runner_processes"] for r in rows), default=0)),
            },
            errors=dict(self._errors),
            notes=notes,
        )


# ---------------------------------------------------------------------- environment
def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def is_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _powershell(command: str, timeout_s: float = 20.0) -> str | None:
    """Ask the Windows host a read-only question from inside WSL. ``None`` if that is impossible."""
    exe = shutil.which("powershell.exe")
    if exe is None:
        return None
    try:
        done = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            timeout=timeout_s,
            check=False,
            cwd="/tmp",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip() or None


def host_facts() -> dict[str, Any]:
    """Facts about the physical host that the Linux guest cannot see for itself.

    Every value is either read from the host or reported as ``None`` with the reason; nothing is
    assumed. Outside WSL the Linux view *is* the host, and this says so.
    """
    if not is_wsl():
        return {
            "virtualised": False,
            "physical_ram_bytes": psutil.virtual_memory().total,
            "physical_ram_source": "psutil (not running under WSL; the OS view is the host)",
            "wslconfig": None,
        }
    facts: dict[str, Any] = {
        "virtualised": True,
        "physical_ram_bytes": None,
        "physical_ram_source": "unavailable: powershell.exe could not be queried from WSL",
        "wslconfig": None,
        "wslconfig_source": "unavailable",
    }
    raw = _powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
    if raw and raw.splitlines()[-1].strip().isdigit():
        facts["physical_ram_bytes"] = int(raw.splitlines()[-1].strip())
        facts["physical_ram_source"] = "Windows host, Win32_ComputerSystem.TotalPhysicalMemory"
    config = _powershell(
        "if (Test-Path $env:USERPROFILE\\.wslconfig) "
        "{ Get-Content $env:USERPROFILE\\.wslconfig -Raw } else { 'ABSENT' }"
    )
    if config is not None:
        facts["wslconfig_source"] = "Windows host, %USERPROFILE%\\.wslconfig"
        facts["wslconfig"] = (
            "absent (WSL2 defaults apply: half of host memory, all logical processors)"
            if config.strip() == "ABSENT"
            else config.strip()[:2000]
        )
    return facts


def environment_info(include_host: bool = True) -> dict[str, Any]:
    """Hardware and OS as this process sees them, plus the host's view where it can be had.

    Memory is given in bytes and in both GiB (2**30, what ``free -h`` prints) and GB (10**9),
    because quoting one against the other is how "7.7" and "8.2" came to describe one machine.
    """
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    cpu_model = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model or platform.processor() or platform.machine(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "guest_memory_total_bytes": memory.total,
        "guest_memory_total_gib": round(memory.total / 2**30, 2),
        "guest_memory_total_gb_decimal": round(memory.total / 1e9, 2),
        "guest_swap_total_gib": round(swap.total / 2**30, 2),
        "is_wsl": is_wsl(),
        "limitations": [],
    }
    if info["is_wsl"]:
        info["limitations"] += [
            "Running under WSL2: 'guest' memory is the VM's allocation, not the physical host's.",
            "WSL2 returns freed guest memory to Windows lazily, and the Linux page cache "
            "counts toward the VM's footprint on the host; system-wide 'used' excludes cache.",
            "Windows-side memory pressure is invisible from inside the guest.",
        ]
    if include_host:
        host = host_facts()
        info["host"] = host
        if host.get("physical_ram_bytes"):
            info["host"]["physical_ram_gib"] = round(host["physical_ram_bytes"] / 2**30, 2)
    return info
