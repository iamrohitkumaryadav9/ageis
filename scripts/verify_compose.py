"""Verify the Docker Compose stack for real, and record what happened.

    uv run python -m scripts.verify_compose                 # default services (+ tracing)
    uv run python -m scripts.verify_compose --llm           # also the `llm` profile (Ollama in a container)
    uv run python -m scripts.verify_compose --llm --clean   # from empty volumes (re-downloads the model)

Needs AEGIS_API_TOKENS in the environment or in `.env`, with approver/operator/viewer tokens.
Writes ``results/release/compose[-llm].json``. Exit code 0 only if every check passed.

With --llm the script: starts the profile, waits for the one-shot model pull, checks that the API
reaches the in-network Ollama, runs one investigation with the LLM through the API, stops the
Ollama container and checks that an investigation still completes on the deterministic path, then
starts it again and checks that the LLM is usable once more.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

API = "http://localhost:8000"
DASHBOARD = "http://localhost:8080"


def sh(*args: str, timeout: float = 900, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def compose(*args: str, profiles: list[str], timeout: float = 3600) -> subprocess.CompletedProcess:
    flags = [flag for p in profiles for flag in ("--profile", p)]
    return sh("docker", "compose", *flags, *args, timeout=timeout)


def tokens() -> dict[str, str]:
    raw = os.environ.get("AEGIS_API_TOKENS", "")
    if not raw and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("AEGIS_API_TOKENS="):
                raw = line.split("=", 1)[1].strip()
    out = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3:
            out[parts[2]] = parts[0]
    missing = {"approver", "operator", "viewer"} - set(out)
    if missing:
        sys.exit(f"AEGIS_API_TOKENS must define tokens for roles: {sorted(missing)}")
    return out


def wait_for(predicate, timeout: float, interval: float = 3.0) -> tuple[bool, float]:
    started = time.time()
    while time.time() - started < timeout:
        try:
            if predicate():
                return True, time.time() - started
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        time.sleep(interval)
    return False, time.time() - started


def ready() -> dict[str, Any]:
    return httpx.get(f"{API}/ready", timeout=10).json()


def services(profiles: list[str]) -> dict[str, str]:
    out = compose("ps", "-a", "--format", "{{.Service}}\t{{.Status}}", profiles=profiles).stdout
    return dict(line.split("\t", 1) for line in out.strip().splitlines() if "\t" in line)


def investigate(operator: str, use_llm: bool, timeout: float) -> dict[str, Any]:
    """Seed an incident through the API (deterministically), then investigate it again."""
    seeded = sh(
        sys.executable, "-m", "scripts.seed_incident", "--operator-token", operator, timeout=600
    )
    if seeded.returncode != 0:
        return {"error": f"seeding failed: {(seeded.stderr or seeded.stdout)[-300:]}"}
    seed = json.loads(seeded.stdout.strip().splitlines()[-1])
    headers = {"Authorization": f"Bearer {operator}"}
    started = time.time()
    response = httpx.post(
        f"{API}/incidents/{seed['incident_id']}/investigate",
        json={"use_llm": use_llm, "wait": True},
        headers=headers,
        timeout=timeout,
    )
    if response.status_code >= 400:
        return {"error": f"investigate -> {response.status_code}: {response.text[:300]}"}
    detail = httpx.get(
        f"{API}/investigations/{response.json()['id']}", headers=headers, timeout=30
    ).json()
    top = (detail.get("hypotheses") or [{}])[0]
    return {
        "incident_id": seed["incident_id"],
        "status": detail.get("status"),
        "llm_status": detail.get("llm_status"),
        "llm_model": detail.get("llm_model"),
        "seconds": round(time.time() - started, 1),
        "llm_calls": [
            {
                "purpose": c["purpose"],
                "ok": c["ok"],
                "error": c["error"],
                "seconds": round(c["duration_ms"] / 1000, 1),
            }
            for c in detail.get("llm_calls", [])
        ],
        "top_cause": top.get("root_cause"),
        "ranked_by": top.get("ranked_by"),
        "errors": detail.get("errors", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--clean", action="store_true", help="remove volumes first")
    parser.add_argument("--keep", action="store_true", help="leave the stack running afterwards")
    parser.add_argument("--pull-timeout", type=float, default=3600)
    parser.add_argument("--llm-timeout", type=float, default=900)
    parser.add_argument("--out", type=Path, default=Path("results/release"))
    args = parser.parse_args()

    tok = tokens()
    profiles = ["tracing", *(["llm"] if args.llm else [])]
    os.environ.setdefault("AEGIS_OTLP_ENDPOINT", "http://jaeger:4318")
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, /, **detail: Any) -> None:
        checks[name] = {"passed": bool(passed), **detail}
        print(
            f"  {'ok  ' if passed else 'FAIL'}  {name}  {json.dumps(detail, default=str)[:260]}",
            flush=True,
        )

    config = compose("config", "-q", profiles=profiles)
    record("compose_config_valid", config.returncode == 0, stderr=config.stderr[-300:])
    if args.clean:
        compose("down", "-v", "--remove-orphans", profiles=["tracing", "llm"])
    started = time.time()
    up = compose("up", "-d", profiles=profiles, timeout=3600)
    record(
        "compose_up",
        up.returncode == 0,
        seconds=round(time.time() - started, 1),
        stderr=up.stderr[-400:] if up.returncode else "",
    )
    ok, waited = wait_for(lambda: ready()["status"] == "ready", 600)
    state = ready() if ok else {}
    record(
        "api_ready",
        ok,
        waited_s=round(waited, 1),
        components={k: v["status"] for k, v in state.get("components", {}).items()},
    )
    record("total_startup", ok, seconds=round(time.time() - started, 1), clean_volumes=args.clean)

    dash = httpx.get(DASHBOARD, timeout=10)
    proxied = httpx.get(
        f"{DASHBOARD}/api/overview",
        timeout=10,
        headers={"Authorization": f"Bearer {tok['viewer']}"},
    )
    record(
        "dashboard_serves_and_proxies",
        dash.status_code == 200
        and proxied.status_code == 200
        and httpx.get(f"{DASHBOARD}/api/overview", timeout=10).status_code == 401,
        index=dash.status_code,
        proxied_overview=proxied.status_code,
    )
    mlflow_ok, _ = wait_for(
        lambda: httpx.get("http://localhost:5000/health", timeout=5).status_code == 200, 120
    )
    experiments = []
    if mlflow_ok:
        experiments = [
            e["name"]
            for e in httpx.post(
                "http://localhost:5000/api/2.0/mlflow/experiments/search",
                json={"max_results": 20},
                timeout=10,
            )
            .json()
            .get("experiments", [])
        ]
    record(
        "mlflow_lists_training_runs",
        mlflow_ok and "aegis-baseline-adult" in experiments,
        experiments=experiments,
    )
    httpx.get(f"{API}/overview", headers={"Authorization": f"Bearer {tok['viewer']}"}, timeout=10)
    traced, _ = wait_for(
        lambda: (
            "aegis"
            in (
                httpx.get("http://localhost:16686/api/services", timeout=5).json().get("data") or []
            )
        ),
        60,
    )
    record("traces_reach_jaeger", traced)

    base = investigate(tok["operator"], use_llm=False, timeout=300)
    record(
        "deterministic_investigation",
        base.get("status") == "completed" and base.get("ranked_by") == "rules",
        **base,
    )

    if args.llm:

        def pulled() -> bool:
            status = services(profiles).get("ollama-pull", "")
            if status.startswith("Exited") and "Exited (0)" not in status:
                raise SystemExit(
                    f"ollama-pull failed: {status}\n"
                    + compose("logs", "--tail", "30", "ollama-pull", profiles=profiles).stdout[
                        -1500:
                    ]
                )
            return "Exited (0)" in status

        pull_started = time.time()
        ok, waited = wait_for(pulled, args.pull_timeout, interval=10)
        listing = compose("exec", "-T", "ollama", "ollama", "list", profiles=profiles).stdout
        record(
            "model_pulled_in_container",
            ok,
            waited_s=round(time.time() - pull_started, 1),
            ollama_list=listing.strip().splitlines(),
        )
        health = services(profiles)
        record(
            "ollama_service_healthy",
            "healthy" in health.get("ollama", ""),
            status=health.get("ollama"),
        )
        ok, waited = wait_for(lambda: ready()["components"]["llm"]["status"] == "ok", 120)
        record(
            "api_reaches_ollama_over_compose_network",
            ok,
            llm=ready()["components"]["llm"],
            endpoint="http://ollama:11434 (service name, no host port)",
        )
        # `docker compose port` prints ":0" (or nothing, or an error) for an unpublished port.
        published = compose("port", "ollama", "11434", profiles=profiles).stdout.strip()
        host_port = published.rsplit(":", 1)[-1] if published else ""
        record(
            "ollama_not_published_to_host",
            host_port in ("", "0"),
            port_output=published,
            note="the model server is reachable only from inside the Compose network",
        )

        with_llm = investigate(tok["operator"], use_llm=True, timeout=args.llm_timeout)
        record(
            "llm_investigation_through_api",
            with_llm.get("status") == "completed"
            and any(c["ok"] for c in with_llm.get("llm_calls", [])),
            **with_llm,
        )

        compose("stop", "ollama", profiles=profiles)
        ok, _ = wait_for(lambda: ready()["components"]["llm"]["status"] != "ok", 60)
        degraded = ready()
        record(
            "ready_reports_llm_unavailable_but_system_ready",
            ok and degraded["status"] == "ready",
            llm=degraded["components"]["llm"],
            system=degraded["status"],
        )
        offline = investigate(tok["operator"], use_llm=True, timeout=300)
        record(
            "fallback_to_deterministic_when_ollama_is_down",
            offline.get("status") == "completed"
            and offline.get("llm_status") == "unavailable"
            and offline.get("ranked_by") == "rules",
            **offline,
        )

        compose("start", "ollama", profiles=profiles)
        ok, waited = wait_for(lambda: ready()["components"]["llm"]["status"] == "ok", 180)
        record(
            "llm_recovers_after_ollama_restart",
            ok,
            waited_s=round(waited, 1),
            llm=ready()["components"]["llm"],
        )

    status = services(profiles)
    logs = {
        name: compose("logs", "--tail", "15", name, profiles=profiles).stdout[-1500:]
        for name in ("api", *(["ollama", "ollama-pull"] if args.llm else []))
    }
    unhealthy = {
        k: v
        for k, v in status.items()
        if not (v.startswith("Up") or (k == "ollama-pull" and "Exited (0)" in v))
    }
    record("all_services_up", not unhealthy, services=status, problems=unhealthy)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("compose-llm.json" if args.llm else "compose.json")
    path.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "profiles": profiles,
                "clean_volumes": args.clean,
                "passed": all(c["passed"] for c in checks.values()),
                "checks": checks,
                "log_tails": logs,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    if not args.keep:
        compose("down", profiles=["tracing", "llm"])
    failed = [n for n, c in checks.items() if not c["passed"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed -> {path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
