"""Create one real incident with a validated, pending remediation, over the public HTTP API.

Used by the browser verification (`apps/dashboard/e2e/browser-check.mjs`) and handy for a manual
look at the dashboard. Nothing is written to the database directly.

    uv run python -m scripts.seed_incident --operator-token <t> [--api http://localhost:8000]

Prints one JSON object: ``{"incident_id": ..., "investigation_id": ..., "remediation_id": ...}``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import httpx

from aegis.config import get_settings
from aegis.faults.scenarios import build_scenario
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--operator-token", required=True)
    parser.add_argument("--dataset", default="adult", choices=sorted(DATASETS))
    parser.add_argument("--fault", default="F04")
    parser.add_argument("--severity", default="high")
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    spec = DATASETS[args.dataset]
    registry = ModelRegistry(settings.model_dir)
    scenario = build_scenario(
        spec,
        read_split(spec.name, "production", settings.processed_dir),
        registry,
        registry.baseline(spec.name).version,
        fault_id=args.fault,
        severity=args.severity,
        seed=args.seed,
        n_rows=2000,
        workdir=Path(tempfile.mkdtemp(prefix="aegis-seed-")),
    )
    client = httpx.Client(
        base_url=args.api, headers={"Authorization": f"Bearer {args.operator_token}"}, timeout=300
    )

    def check(response: httpx.Response) -> dict:
        if response.status_code >= 400:
            sys.exit(
                f"{response.request.method} {response.request.url.path} -> "
                f"{response.status_code}: {response.text[:300]}"
            )
        return response.json()

    buffer = io.StringIO()
    scenario.current.to_csv(buffer, index=False)
    batch = check(
        client.post(
            "/data/current",
            data={"dataset": spec.name},
            files={"file": ("batch.csv", buffer.getvalue().encode(), "text/csv")},
        )
    )
    monitored = check(
        client.post("/monitor/run", json={"dataset": spec.name, "batch_id": batch["id"]})
    )
    incident = monitored["incident"]
    if incident is None:
        sys.exit("monitoring opened no incident for this batch")
    investigation = check(
        client.post(
            f"/incidents/{incident['id']}/investigate", json={"use_llm": False, "wait": True}
        )
    )
    detail = check(client.get(f"/incidents/{incident['id']}"))
    remediations = detail.get("remediations") or []
    print(
        json.dumps(
            {
                "incident_id": incident["id"],
                "investigation_id": investigation["id"],
                "remediation_id": remediations[0]["id"] if remediations else None,
                "remediation_state": remediations[0]["state"] if remediations else None,
                "feature": scenario.truth.affected_features[0],
            }
        )
    )


if __name__ == "__main__":
    main()
