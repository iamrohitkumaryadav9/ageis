"""One reproducible release verification: run every gate, record what actually happened.

    uv run python -m scripts.release_verify                  # everything
    uv run python -m scripts.release_verify --skip-optional  # mandatory gates only
    uv run python -m scripts.release_verify --only frontend_tests,frontend_build
    uv run python -m scripts.release_verify --list

Writes ``results/release/release-report.json`` and a log per gate under
``results/release/logs/``. Exit code 0 when every mandatory gate passed.

Gate statuses
-------------
passed    the command ran and succeeded
failed    the command ran and did not succeed
blocked   a prerequisite is missing (Docker, Node, a browser, a model, a token): NOT a pass
skipped   excluded on the command line: NOT a pass

Mandatory gates decide the verdict. Optional gates need hardware or downloads this project
cannot assume (a local LLM, several GB of RAM); a failed or blocked optional gate downgrades the
verdict to "conditionally ready" and is listed, never hidden.

Prerequisites: PostgreSQL for the tests (``make test-db-up``), Docker, Node (PATH or
``.local/node``), ``AEGIS_API_TOKENS`` in the environment or ``.env`` for the Compose gates, and
for the optional gates a running Ollama with the models pulled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "release"
DASHBOARD = ROOT / "apps" / "dashboard"
REFERENCE_RUN = "main"
REFERENCE_SHA256 = "910170c1f1119b6d119b7b361767fcf4b0c8d32d43d3ccde94767c9d715374f5"
REFERENCE_HEADLINE = {"overall_fault_detection_f1": 0.9958, "drift_detection_f1": 0.9565}
PY = [sys.executable, "-m"]
PYTEST = [*PY, "pytest", "-q", "-p", "no:cacheprovider"]


@dataclass
class Gate:
    id: str
    title: str
    mandatory: bool
    run: Any  # callable(ctx) -> GateResult
    needs: tuple[str, ...] = ()


@dataclass
class GateResult:
    status: str
    detail: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    log: str = ""


class Context:
    def __init__(self) -> None:
        self.env = dict(os.environ)
        node = ROOT / ".local" / "node" / "bin"
        if node.is_dir():
            self.env["PATH"] = f"{node}{os.pathsep}{self.env.get('PATH', '')}"
        libs = ROOT / ".local" / "browser-libs" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
        if libs.is_dir():
            self.env["LD_LIBRARY_PATH"] = f"{libs}{os.pathsep}{self.env.get('LD_LIBRARY_PATH', '')}"
        browsers = ROOT / ".local" / "ms-playwright"
        if browsers.is_dir():
            self.env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(browsers))
        self.env.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        self.tokens = self._tokens()
        self.compose_up = False

    def _tokens(self) -> dict[str, str]:
        raw = self.env.get("AEGIS_API_TOKENS", "")
        if not raw and (ROOT / ".env").exists():
            for line in (ROOT / ".env").read_text().splitlines():
                if line.startswith("AEGIS_API_TOKENS="):
                    raw = line.split("=", 1)[1].strip()
                    self.env["AEGIS_API_TOKENS"] = raw
        out = {}
        for entry in raw.split(","):
            parts = entry.strip().split(":")
            if len(parts) == 3 and not parts[0].startswith("CHANGE-ME"):
                out[parts[2]] = parts[0]
        return out

    def has(self, need: str) -> str | None:
        """Return None if the prerequisite is met, else the reason it is not."""
        if need == "docker":
            if not shutil.which("docker"):
                return "docker is not installed"
            done = subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False)
            return None if done.returncode == 0 else "the Docker daemon is not reachable"
        if need == "node":
            return None if shutil.which("npm", path=self.env["PATH"]) else "npm not found"
        if need == "tokens":
            missing = {"approver", "operator", "viewer"} - set(self.tokens)
            return f"AEGIS_API_TOKENS lacks roles {sorted(missing)}" if missing else None
        if need == "postgres":
            done = self.cmd(
                [*PY, "pytest", "tests/integration/test_api.py", "--collect-only", "-q"], 120
            )
            return None if done.returncode == 0 else "integration tests cannot be collected"
        if need == "compose_stack":
            return (
                None if self.compose_up else "the Compose stack gate did not leave a running stack"
            )
        if need.startswith("ollama:"):
            return self._ollama(need.split(":", 1)[1])
        return f"unknown prerequisite {need}"

    def _ollama(self, model: str) -> str | None:
        import httpx

        from aegis.config import get_settings

        base = get_settings().ollama_base_url
        try:
            names = [m["name"] for m in httpx.get(f"{base}/api/tags", timeout=5).json()["models"]]
        except Exception as exc:
            return f"Ollama is not reachable at {base} ({type(exc).__name__})"
        return None if model in names else f"model {model} is not pulled (ollama pull {model})"

    def cmd(self, args: list[str], timeout: float, cwd: Path = ROOT, env: dict | None = None):
        return subprocess.run(
            args,
            cwd=cwd,
            env={**self.env, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


def _tail(text: str, lines: int = 6) -> list[str]:
    return [line[:300] for line in text.strip().splitlines()[-lines:]]


def command(args: list[str], *, timeout: float = 1800, cwd: Path = ROOT, parse=None, env=None):
    def run(ctx: Context) -> GateResult:
        try:
            done = ctx.cmd(args, timeout, cwd, env)
        except subprocess.TimeoutExpired as exc:
            output = (
                (exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else ""
            )
            return GateResult("failed", {"error": f"timed out after {timeout:.0f}s"}, log=output)
        output = done.stdout + ("\n--- stderr ---\n" + done.stderr if done.stderr.strip() else "")
        detail: dict[str, Any] = {"command": " ".join(args), "exit_code": done.returncode}
        if parse:
            detail.update(parse(done.stdout))
        detail["output_tail"] = _tail(done.stdout or done.stderr)
        return GateResult("passed" if done.returncode == 0 else "failed", detail, log=output)

    return run


def pytest_counts(stdout: str) -> dict[str, Any]:
    import re

    summary = next(
        (
            line
            for line in reversed(stdout.splitlines())
            if " passed" in line or " failed" in line or " error" in line
        ),
        "",
    )
    counts = {
        key: int(n)
        for n, key in re.findall(
            r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed)", summary
        )
    }
    return {"tests": counts, "summary_line": summary.strip()[:200]}


def vitest_counts(stdout: str) -> dict[str, Any]:
    import re

    clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    tests = re.search(r"Tests\s+(?:(\d+) failed \| )?(\d+) passed", clean)
    return {
        "tests": {"passed": int(tests.group(2)), "failed": int(tests.group(1) or 0)}
        if tests
        else {}
    }


def json_report(path: Path, keys: tuple[str, ...] = ("passed",)):
    def parse(_: str) -> dict[str, Any]:
        if not path.exists():
            return {"report": "not written"}
        data = json.loads(path.read_text())
        out = {k: data.get(k) for k in keys}
        if "checks" in data and isinstance(data["checks"], dict):
            out["checks"] = {name: c.get("passed") for name, c in data["checks"].items()}
        out["report"] = str(path.relative_to(ROOT))
        return out

    return parse


def benchmark_integrity(ctx: Context) -> GateResult:
    """Raw results unmodified; headline metrics recompute to the published values; lifecycle
    reconciles; the committed summary equals a fresh recomputation."""
    from aegis.evaluation import report as R

    run = ROOT / "results" / "aegis-ir" / REFERENCE_RUN
    detail: dict[str, Any] = {"run": REFERENCE_RUN}
    problems: list[str] = []
    digest = hashlib.sha256((run / "scenarios.jsonl").read_bytes()).hexdigest()
    detail["raw_sha256"] = digest
    if digest != REFERENCE_SHA256:
        problems.append("raw scenarios.jsonl of the reference run differs from its pinned hash")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / REFERENCE_RUN
        shutil.copytree(run, copy)
        fresh = R.summarise(copy)
    headline = fresh["headline_metrics"]
    detail["recomputed"] = {k: headline[k] for k in REFERENCE_HEADLINE}
    detail["populations"] = {
        name: fresh["detection"][name]["population"]["n_positive_scenarios"]
        for name in R.DETECTION_METRICS
    }
    for key, expected in REFERENCE_HEADLINE.items():
        if headline[key] != expected:
            problems.append(f"{key} recomputes to {headline[key]}, published {expected}")
    committed = json.loads((run / "summary.json").read_text())
    if committed.get("headline_metrics") != headline:
        problems.append("committed summary.json is stale: regenerate with aegis.evaluation.report")
    lifecycle = fresh["full_grid"]["B"]["remediation"]["lifecycle"]
    detail["lifecycle_reconciled"] = lifecycle["reconciled"]
    if not lifecycle["reconciled"]:
        problems.append(f"lifecycle reconciliation failed: {lifecycle['failure_counts']}")
    others = {}
    for path in sorted((ROOT / "results" / "aegis-ir").iterdir()):
        if path.name == REFERENCE_RUN or not (path / "scenarios.jsonl").exists():
            continue
        if not (path / "manifest.json").exists() or path.name.startswith("smoke"):
            continue
        summary = R.summarise(path)
        rem = (summary["full_grid"]["B"] or {}).get("remediation") or {}
        ok = (rem.get("lifecycle") or {}).get("reconciled")
        others[path.name] = {
            "lifecycle_reconciled": ok,
            **{k: summary["headline_metrics"][k] for k in REFERENCE_HEADLINE},
        }
        if ok is False:
            problems.append(f"run {path.name}: lifecycle reconciliation failed")
        stored = path / "summary.json"
        if (
            stored.exists()
            and json.loads(stored.read_text()).get("headline_metrics")
            != summary["headline_metrics"]
        ):
            problems.append(f"run {path.name}: committed summary.json is stale")
    detail["other_runs"] = others
    detail["problems"] = problems
    tests = command(
        [
            *PYTEST,
            "tests/unit/test_evaluation_metrics.py",
            "tests/unit/test_lifecycle_reconciliation.py",
        ],
        parse=pytest_counts,
    )(ctx)
    detail["tests"] = tests.detail.get("tests")
    if tests.status != "passed":
        problems.append("metric / lifecycle tests failed")
    return GateResult("failed" if problems else "passed", detail, log=tests.log)


def compose_default(ctx: Context) -> GateResult:
    result = command(
        [*PY, "scripts.verify_compose", "--keep"],
        timeout=2400,
        parse=json_report(OUT / "compose.json"),
    )(ctx)
    ctx.compose_up = result.status == "passed"
    return result


def browser(ctx: Context) -> GateResult:
    seeded = ctx.cmd(
        [*PY, "scripts.seed_incident", "--operator-token", ctx.tokens["operator"]], 600
    )
    if seeded.returncode != 0:
        return GateResult(
            "failed",
            {
                "error": "could not seed an incident",
                "output_tail": _tail(seeded.stderr or seeded.stdout),
            },
            log=seeded.stdout + seeded.stderr,
        )
    incident = json.loads(seeded.stdout.strip().splitlines()[-1])["incident_id"]
    probe = ctx.cmd(
        [
            "node",
            "-e",
            "import('playwright').then(async p=>{const b=await p.chromium.launch();await b.close()})",
        ],
        120,
        DASHBOARD,
    )
    if probe.returncode != 0:
        return GateResult(
            "blocked",
            {
                "reason": "headless Chromium cannot start in this environment",
                "how_to_fix": "cd apps/dashboard && npx playwright install --with-deps chromium-headless-shell "
                "(needs root for system libraries), or see docs/release-readiness.md for a rootless setup",
                "output_tail": _tail(probe.stderr),
            },
            log=probe.stderr,
        )
    return command(
        ["node", "e2e/browser-check.mjs"],
        cwd=DASHBOARD,
        timeout=900,
        parse=json_report(
            OUT / "browser" / "browser-report.json",
            ("passed", "checks_total", "checks_failed", "browser", "viewports"),
        ),
        env={
            "DASHBOARD_URL": "http://localhost:8080",
            "VIEWER_TOKEN": ctx.tokens["viewer"],
            "OPERATOR_TOKEN": ctx.tokens["operator"],
            "APPROVER_TOKEN": ctx.tokens["approver"],
            "INCIDENT_ID": incident,
            "BROWSER_REPORT_DIR": str(OUT / "browser"),
        },
    )(ctx)


def compose_down(ctx: Context) -> None:
    ctx.cmd(["docker", "compose", "--profile", "tracing", "--profile", "llm", "down"], 300)
    ctx.compose_up = False


def llm_report(model: str) -> Path:
    return OUT / f"llm-{model.replace(':', '-')}.json"


GATES: list[Gate] = [
    Gate(
        "lint",
        "Static checks (ruff check, ruff format --check)",
        True,
        lambda ctx: _both(
            ctx, [*PY, "ruff", "check", "."], [*PY, "ruff", "format", "--check", "."]
        ),
    ),
    Gate(
        "backend_unit",
        "Backend unit tests",
        True,
        command([*PYTEST, "tests/unit"], parse=pytest_counts),
    ),
    Gate(
        "postgres_integration",
        "PostgreSQL integration tests (API, pipeline)",
        True,
        command(
            [*PYTEST, "tests/integration/test_api.py", "tests/integration/test_pipeline.py"],
            parse=pytest_counts,
        ),
        needs=("postgres",),
    ),
    Gate(
        "mlflow_tracking",
        "MLflow tracking tests (SQLite store and a local tracking server)",
        True,
        command([*PYTEST, "tests/integration/test_tracking.py"], parse=pytest_counts),
    ),
    Gate(
        "resource_sampler",
        "Resource sampler tests (fake trees and the real process table)",
        True,
        command(
            [
                *PYTEST,
                "tests/unit/test_resources.py",
                "tests/integration/test_resources_live.py",
                "-m",
                "not ollama",
            ],
            parse=pytest_counts,
        ),
    ),
    Gate(
        "adversarial",
        "Adversarial tests (prompt injection, claims, approval bypass)",
        True,
        command([*PYTEST, "tests/adversarial"], parse=pytest_counts),
        needs=("postgres",),
    ),
    Gate(
        "security_policy",
        "Security and policy checks (approval enforcement, tool allowlist, policy, lifecycle)",
        True,
        command(
            [
                *PYTEST,
                "tests/adversarial/test_approval_adversarial.py",
                "tests/unit/test_tools.py",
                "tests/unit/test_remediation.py",
                "tests/unit/test_lifecycle_reconciliation.py",
            ],
            parse=pytest_counts,
        ),
        needs=("postgres",),
    ),
    Gate(
        "e2e",
        "End-to-end lifecycle through the HTTP API (deterministic path)",
        True,
        command([*PYTEST, "tests/e2e", "-m", "not ollama"], parse=pytest_counts),
        needs=("postgres",),
    ),
    Gate(
        "benchmark_integrity",
        "Benchmark integrity (raw hash, recomputed metrics, reconciliation)",
        True,
        benchmark_integrity,
    ),
    Gate(
        "frontend_tests",
        "Frontend tests (Vitest, Testing Library)",
        True,
        command(["npm", "test", "--silent"], cwd=DASHBOARD, timeout=600, parse=vitest_counts),
        needs=("node",),
    ),
    Gate(
        "frontend_build",
        "Frontend type-check and production build",
        True,
        command(["npm", "run", "build", "--silent"], cwd=DASHBOARD, timeout=600),
        needs=("node",),
    ),
    Gate(
        "compose_default",
        "Docker Compose: default services + tracing (API, dashboard, MLflow, Jaeger)",
        True,
        compose_default,
        needs=("docker", "tokens"),
    ),
    Gate(
        "browser",
        "Dashboard in a real browser (interactions, error states, phone viewport)",
        True,
        browser,
        needs=("node", "tokens", "compose_stack"),
    ),
    Gate(
        "ollama_qwen3_4b",
        "Local LLM qwen3:4b: structured output, full investigation, failure handling",
        False,
        command(
            [*PY, "scripts.verify_llm", "--model", "qwen3:4b"],
            timeout=3600,
            parse=json_report(llm_report("qwen3:4b")),
        ),
        needs=("ollama:qwen3:4b",),
    ),
    Gate(
        "ollama_llama3_2_3b",
        "Fallback LLM llama3.2:3b: structured output, full investigation, failure handling",
        False,
        command(
            [*PY, "scripts.verify_llm", "--model", "llama3.2:3b"],
            timeout=3600,
            parse=json_report(llm_report("llama3.2:3b")),
        ),
        needs=("ollama:llama3.2:3b",),
    ),
    Gate(
        "e2e_real_llm",
        "End-to-end lifecycle with the real local model",
        False,
        command(
            [*PYTEST, "tests/e2e", "tests/integration/test_resources_live.py", "-m", "ollama"],
            timeout=3600,
            parse=pytest_counts,
            env={"AEGIS_OLLAMA_TIMEOUT_S": "300"},
        ),
        needs=("postgres", "ollama:qwen3:4b"),
    ),
    Gate(
        "compose_llm",
        "Docker Compose `llm` profile: in-container Ollama, connectivity, fallback, recovery",
        False,
        command(
            [*PY, "scripts.verify_compose", "--llm"],
            timeout=7200,
            parse=json_report(OUT / "compose-llm.json"),
        ),
        needs=("docker", "tokens"),
    ),
]


def _both(ctx: Context, first: list[str], second: list[str]) -> GateResult:
    a, b = command(first)(ctx), command(second)(ctx)
    status = "passed" if a.status == b.status == "passed" else "failed"
    return GateResult(status, {"check": a.detail, "format": b.detail}, log=a.log + "\n" + b.log)


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_files": len(
            [line for line in git("status", "--porcelain").splitlines() if line]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated gate ids")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--keep-stack", action="store_true", help="leave the Compose stack running")
    args = parser.parse_args()
    if args.list:
        for gate in GATES:
            print(
                f"{'mandatory' if gate.mandatory else 'optional ':<9}  {gate.id:<22} {gate.title}"
            )
        return
    only = {g.strip() for g in args.only.split(",") if g.strip()}
    unknown = only - {g.id for g in GATES}
    if unknown:
        sys.exit(f"unknown gate ids: {sorted(unknown)}")

    ctx = Context()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.UTC)
    results: list[dict[str, Any]] = []
    try:
        for gate in GATES:
            entry: dict[str, Any] = {
                "id": gate.id,
                "title": gate.title,
                "mandatory": gate.mandatory,
            }
            if (only and gate.id not in only) or (args.skip_optional and not gate.mandatory):
                entry.update(
                    status="skipped", detail={"reason": "excluded on the command line"}, seconds=0.0
                )
            else:
                if gate.id == "compose_llm" and ctx.compose_up:
                    compose_down(ctx)  # the llm profile check starts and stops its own stack
                reason = next((r for r in (ctx.has(n) for n in gate.needs) if r), None)
                if reason:
                    entry.update(status="blocked", detail={"reason": reason}, seconds=0.0)
                else:
                    print(f"[{gate.id}] {gate.title} ...", flush=True)
                    tick = time.perf_counter()
                    try:
                        outcome = gate.run(ctx)
                    except Exception as exc:
                        outcome = GateResult(
                            "failed", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                        )
                    outcome.seconds = round(time.perf_counter() - tick, 1)
                    # last 400 lines: enough to see a failure, small enough to commit
                    tail = "\n".join((outcome.log or "").splitlines()[-400:])
                    (OUT / "logs" / f"{gate.id}.txt").write_text(tail + "\n")
                    entry.update(
                        status=outcome.status,
                        detail=outcome.detail,
                        seconds=outcome.seconds,
                        log=f"results/release/logs/{gate.id}.txt",
                    )
            print(f"  -> {entry['status'].upper():<8} {gate.id} ({entry['seconds']}s)", flush=True)
            results.append(entry)
    finally:
        if ctx.compose_up and not args.keep_stack:
            compose_down(ctx)

    mandatory = [r for r in results if r["mandatory"]]
    optional = [r for r in results if not r["mandatory"]]
    mandatory_ok = all(r["status"] == "passed" for r in mandatory)
    optional_ok = all(r["status"] == "passed" for r in optional)
    verdict = (
        ("RELEASE READY" if optional_ok else "CONDITIONALLY READY") if mandatory_ok else "NOT READY"
    )
    partial = bool(only) or args.skip_optional
    report = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "started_at": started_at.isoformat(timespec="seconds"),
        "duration_s": round((dt.datetime.now(dt.UTC) - started_at).total_seconds(), 1),
        "git": git_state(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "partial_run": partial,
        "verdict": verdict
        if not partial
        else f"{verdict} (PARTIAL RUN: skipped gates are not passes)",
        "mandatory": {"total": len(mandatory), **_counts(mandatory)},
        "optional": {"total": len(optional), **_counts(optional)},
        "gates": results,
    }
    (OUT / "release-report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (OUT / "release-report.md").write_text(markdown(report))
    print(f"\n{'gate':<22} {'kind':<9} {'status':<8} seconds")
    for r in results:
        print(
            f"{r['id']:<22} {'mandatory' if r['mandatory'] else 'optional':<9} {r['status']:<8} {r['seconds']}"
        )
    print(f"\nverdict: {report['verdict']}\nreport:  {OUT / 'release-report.json'}")
    sys.exit(0 if mandatory_ok else 1)


def _summary(entry: dict[str, Any]) -> str:
    """One line of evidence for a gate, taken from what the gate itself reported."""
    detail = entry.get("detail") or {}
    if entry["status"] in ("blocked", "skipped"):
        return str(detail.get("reason", ""))
    parts = []
    tests = detail.get("tests")
    if tests:
        parts.append(", ".join(f"{n} {k}" for k, n in tests.items()))
    checks = detail.get("checks")
    if isinstance(checks, dict):
        parts.append(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks")
        failed = [k for k, v in checks.items() if not v]
        if failed:
            parts.append("failed: " + ", ".join(failed))
    if "checks_total" in detail:
        parts.append(
            f"{detail['checks_total'] - detail['checks_failed']}/{detail['checks_total']} browser checks"
        )
    if "recomputed" in detail:
        parts.append(", ".join(f"{k}={v}" for k, v in detail["recomputed"].items()))
        parts.append(f"lifecycle reconciled={detail.get('lifecycle_reconciled')}")
    for problem in detail.get("problems") or []:
        parts.append(f"PROBLEM: {problem}")
    if detail.get("error"):
        parts.append(str(detail["error"]))
    if detail.get("report"):
        parts.append(f"`{detail['report']}`")
    return "; ".join(parts)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Release verification report",
        "",
        "Generated by `python -m scripts.release_verify`. Do not edit by hand.",
        "",
        f"- Generated: {report['generated_at']} (run took {report['duration_s']} s)",
        f"- Commit: `{report['git']['commit'][:12]}` on `{report['git']['branch']}`, "
        f"{report['git']['uncommitted_files']} uncommitted file(s) at the time",
        f"- Platform: {report['platform']}, Python {report['python']}",
        f"- **Verdict: {report['verdict']}**",
        f"- Mandatory gates: {report['mandatory']['passed']}/{report['mandatory']['total']} passed "
        f"({report['mandatory']['failed']} failed, {report['mandatory']['blocked']} blocked, "
        f"{report['mandatory']['skipped']} skipped)",
        f"- Optional gates: {report['optional']['passed']}/{report['optional']['total']} passed "
        f"({report['optional']['failed']} failed, {report['optional']['blocked']} blocked, "
        f"{report['optional']['skipped']} skipped)",
        "",
        "| Gate | Kind | Status | Seconds | Evidence |",
        "|---|---|---|---|---|",
    ]
    for r in report["gates"]:
        status = r["status"] if r["status"] == "passed" else f"**{r['status']}**"
        lines.append(
            f"| {r['title']} (`{r['id']}`) | {'mandatory' if r['mandatory'] else 'optional'} | "
            f"{status} | {r['seconds']} | {_summary(r)} |"
        )
    lines += [
        "",
        "`blocked` and `skipped` are not passes. Logs: `results/release/logs/<gate>.txt`.",
        "",
    ]
    return "\n".join(lines)


def _counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        s: sum(r["status"] == s for r in results)
        for s in ("passed", "failed", "blocked", "skipped")
    }


if __name__ == "__main__":
    main()
