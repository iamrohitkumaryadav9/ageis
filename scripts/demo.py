"""End-to-end demonstration against a running AEGIS API.

A feature's unit silently changes upstream (hours -> minutes). AEGIS detects it, investigates,
proposes and validates a fix, waits for a human, executes on approval, and can roll back.

Usage (API running, see README):
    uv run python -m scripts.demo --operator-token <t> --approver-token <t> [--no-llm] [--fault F04]

Each step prints what the API returned. Nothing is simulated: the batch is uploaded as CSV and
every later step is an HTTP call.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
from pathlib import Path

import httpx

from aegis.config import get_settings
from aegis.faults.scenarios import build_scenario
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--operator-token", required=True)
    parser.add_argument("--approver-token", required=True)
    parser.add_argument("--dataset", default="adult", choices=sorted(DATASETS))
    parser.add_argument("--fault", default="F04")
    parser.add_argument("--severity", default="medium")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-llm", action="store_true", help="use the deterministic investigation path"
    )
    parser.add_argument(
        "--no-rollback", action="store_true", help="leave the fix in place at the end"
    )
    args = parser.parse_args()

    settings = get_settings()
    spec = DATASETS[args.dataset]
    registry = ModelRegistry(settings.model_dir)
    production = read_split(spec.name, "production", settings.processed_dir)
    scenario = build_scenario(
        spec,
        production,
        registry,
        registry.baseline(spec.name).version,
        fault_id=args.fault,
        severity=args.severity,
        seed=args.seed,
        n_rows=2000,
        workdir=Path(tempfile.mkdtemp(prefix="aegis-demo-")),
    )
    if scenario.artifact_path is not None:
        sys.exit(
            "F10 needs a deployment change, which is not exposed over HTTP; pick a data fault for the demo."
        )

    operator = httpx.Client(
        base_url=args.api, headers={"Authorization": f"Bearer {args.operator_token}"}, timeout=900
    )
    approver = httpx.Client(
        base_url=args.api, headers={"Authorization": f"Bearer {args.approver_token}"}, timeout=900
    )

    def step(title: str) -> None:
        print(f"\n=== {title} ===")

    def check(response: httpx.Response) -> dict:
        if response.status_code >= 400:
            sys.exit(
                f"{response.request.method} {response.request.url.path} -> {response.status_code}: {response.text}"
            )
        return response.json()

    step("0. readiness")
    ready = operator.get("/ready").json()
    for name, component in ready["components"].items():
        print(
            f"  {name:<9} {component['status']:<12} {'required' if component['required'] else 'optional'}  {component['detail']}"
        )

    step(
        f"1. upload a production batch (2000 rows; fault injected: {scenario.truth.fault_id}, hidden from AEGIS)"
    )
    buffer = io.StringIO()
    scenario.current.to_csv(buffer, index=False)
    batch = check(
        operator.post(
            "/data/current",
            data={"dataset": spec.name},
            files={"file": ("production_batch.csv", buffer.getvalue().encode(), "text/csv")},
        )
    )
    print(f"  batch {batch['id']}  rows={batch['rows']}  sha256={batch['sha256'][:16]}...")

    step("2. run monitoring")
    monitored = check(
        operator.post("/monitor/run", json={"dataset": spec.name, "batch_id": batch["id"]})
    )
    run, incident = monitored["run"], monitored["incident"]
    print(
        f"  run {run['id']}: incident_detected={run['incident_detected']} severity={run['max_severity']} "
        f"({run['triggered_count']} findings, {run['duration_ms']:.0f} ms)"
    )
    if incident is None:
        sys.exit("  no incident was detected; nothing to investigate")
    print(f"  incident {incident['id']} opened: {incident['summary'][:160]}")

    step(
        f"3. investigate ({'deterministic path' if args.no_llm else 'local LLM; this takes minutes on a CPU'})"
    )
    started = time.time()
    investigation = check(
        operator.post(f"/incidents/{incident['id']}/investigate", json={"use_llm": not args.no_llm})
    )
    while investigation["status"] == "running":
        time.sleep(5)
        investigation = check(operator.get(f"/investigations/{investigation['id']}"))
        print(f"  ... {investigation['status']} ({time.time() - started:.0f}s)", end="\r")
    detail = check(operator.get(f"/investigations/{investigation['id']}"))
    print(
        f"  status={detail['status']} llm_status={detail['llm_status']} model={detail['llm_model']} "
        f"duration={(detail['duration_ms'] or 0) / 1000:.1f}s"
    )
    for call in detail["llm_calls"]:
        print(
            f"    llm {call['purpose']:<8} ok={call['ok']} {call['duration_ms'] / 1000:.1f}s error={call['error']}"
        )
    for trace in detail["tool_traces"]:
        print(
            f"    tool {trace['tool']:<30} {trace['status']:<8} by {trace['requested_by']:<8} {trace['duration_ms']:.0f} ms"
        )
    for note in detail["errors"]:
        print(f"    note: {note}")

    step("4. root-cause hypotheses (ranked by evidence fit; not established causes)")
    for hyp in detail["hypotheses"][:3]:
        print(
            f"  #{hyp['rank']} {hyp['root_cause']:<18} fit={hyp['confidence']:.2f} by={hyp['ranked_by']} "
            f"support={len(hyp['supporting_evidence_ids'])} contra={len(hyp['contradicting_evidence_ids'])}"
        )
    print(f"  conclusion: {(detail['diagnosis'] or {}).get('conclusion')}")
    print(f"  runbooks: {[r['passage_id'] for r in detail['runbook_refs']]}")

    step("5. remediation proposal, sandbox validation, policy")
    remediations = check(operator.get(f"/incidents/{incident['id']}"))["remediations"]
    if not remediations:
        print(
            f"  recommendation: {(detail['recommendation'] or {}).get('action')} - no state change is proposed"
        )
        print(f"  policy: {(detail['policy_decision'] or {}).get('outcome')}")
        return
    remediation = remediations[0]
    print(
        f"  {remediation['id']}: {remediation['action']} {remediation['proposal']['parameters']}  state={remediation['state']}"
    )
    for item in (remediation["validation"] or {}).get("checks", []):
        print(
            f"    [{'pass' if item['passed'] else 'FAIL'}] {item['category']:<12} {item['name']}: {item['detail'][:110]}"
        )
    print(
        f"  policy: {remediation['policy_decision']['outcome']}  {remediation['policy_decision']['reasons']}"
    )
    if remediation["state"] != "PENDING_APPROVAL":
        print("  the proposal did not reach approval; nothing will be executed")
        return

    step("6. approval is enforced by the backend")
    refused = operator.post(
        f"/remediation/{remediation['id']}/approve",
        json={"expected_version": remediation["version"]},
    )
    print(
        f"  operator tries to approve -> HTTP {refused.status_code}: {refused.json()['error']['message']}"
    )
    approved = check(
        approver.post(
            f"/remediation/{remediation['id']}/approve",
            json={
                "expected_version": remediation["version"],
                "comment": "unit change confirmed with data owner",
            },
            headers={"Idempotency-Key": f"demo-{remediation['id']}"},
        )
    )
    print(
        f"  approver approves -> state={approved['state']} executed={approved['executed']} "
        f"deployment revision #{approved['executed_revision']}"
    )
    print(f"  verification: {approved['verification']}")

    step("7. the same batch under the new deployment")
    again = check(
        operator.post("/monitor/run", json={"dataset": spec.name, "batch_id": batch["id"]})
    )
    print(
        f"  incident_detected={again['run']['incident_detected']} (deployment revision #{again['run']['deployment_revision']})"
    )

    step("8. audit trail")
    trail = check(operator.get(f"/audit?incident_id={incident['id']}&limit=200"))["items"]
    for event in trail:
        print(
            f"  {event['at'][11:19]} {event['actor']:<7} {event['action']:<34} {event['outcome']}"
        )

    if not args.no_rollback:
        step("9. rollback (restores the pre-remediation deployment state)")
        rolled = check(
            approver.post(
                f"/remediation/{remediation['id']}/rollback",
                json={"reason": "demo complete; restoring original configuration"},
            )
        )
        print(f"  state={rolled['state']} rollback revision #{rolled['rollback_revision']}")
    print(
        f"\nground truth (never shown to AEGIS): {scenario.truth.fault_id} on {scenario.truth.affected_features} "
        f"{scenario.truth.params}"
    )


if __name__ == "__main__":
    main()
