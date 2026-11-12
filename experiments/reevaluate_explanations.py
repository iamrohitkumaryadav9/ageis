"""Re-validate the LLM explanations stored by a benchmark run with the current validator.

    uv run python -m experiments.reevaluate_explanations --name v2

Why this exists. Run ``v2`` was executed with a claim validator that (a) treated the workflow's
own decision rationale ("Proposed action: no_action due to no performance loss") as a causal
claim about the incident and (b) discarded a whole explanation when one operator check failed.
Both were corrected afterwards. Re-running the LLM (about 80 minutes) is not needed to see the
effect on the *stored* summaries: the evidence of every scenario is reproduced deterministically
and the stored raw summary is validated again. No model is called.

What it cannot do: run ``v2`` did not store the raw operator checks, so only summaries are
re-evaluated. The output says so. It is an offline re-evaluation and is never merged into the
benchmark's own counters.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from aegis.agents.claims import Intervention
from aegis.agents.graph import AgentRuntime, run_investigation
from aegis.agents.llm_tasks import LLMExplanation, review_explanation
from aegis.agents.tools import ToolContext, ToolExecutor
from aegis.config import get_settings
from aegis.evaluation.benchmark import BenchmarkEnv
from aegis.faults.scenarios import scenario_grid, scenario_id
from aegis.ingestion.datasets import DATASETS
from aegis.monitoring.engine import run_monitoring
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.diagnosis import Diagnosis, Evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="v2")
    parser.add_argument("--out", type=Path, default=Path("results/aegis-ir"))
    args = parser.parse_args()
    out = args.out / args.name
    manifest = json.loads((out / "manifest.json").read_text())
    rows = {
        r["scenario_id"]: r
        for r in map(json.loads, (out / "scenarios.jsonl").read_text().splitlines())
    }
    settings = get_settings()
    work = Path(tempfile.mkdtemp(prefix="aegis-reeval-"))
    shutil.copytree(settings.model_dir, work / "models")
    env = BenchmarkEnv.create(settings, work / "models", work / "scenarios", None)
    grid = scenario_grid(
        sorted(DATASETS), manifest["seeds"], manifest["clean_controls_per_dataset"]
    )
    items = []
    for cell in grid:
        sid = scenario_id(
            cell["dataset"], cell["fault_id"], cell["severity"], cell["seed"], cell["n_rows"]
        )
        texts = ((rows[sid]["configs"].get("D") or {}).get("diagnosis") or {}).get("llm_texts")
        if not texts or not texts.get("explanation"):
            continue
        scenario = env.scenario(cell)
        inp = env.monitoring_input(scenario)
        result = run_monitoring(inp)
        state = DeploymentState(dataset=inp.spec.name, model_version=inp.model_version)
        tools = ToolExecutor(ToolContext(inp, result, env.retriever, state))
        final = run_investigation(
            AgentRuntime(
                tools=tools, policy=env.policy, llm=None, declared_version=inp.model_version
            ),
            "reeval",
            "reeval",
        )
        review = review_explanation(
            LLMExplanation(summary=texts["explanation"], operator_checks=[]),
            [Evidence.model_validate(e) for e in final["evidence"]],
            Diagnosis.model_validate(final["rule_diagnosis"]),
            Intervention.from_validation(final.get("proposal"), final.get("validation")),
        )
        items.append(
            {
                "scenario_id": sid,
                "dataset": cell["dataset"],
                "fault_id": cell["fault_id"],
                "summary": texts["explanation"],
                "accepted_in_run": bool(texts.get("explanation_accepted")),
                "summary_accepted_now": review.ok,
                "reason_now": review.reason or None,
                "not_supported_claims_now": [
                    {
                        "text": c.text,
                        "category": c.category.value,
                        "language": c.language.value,
                        "verdict": c.verdict.value,
                    }
                    for c in review.claims
                    if c.verdict.value != "supported"
                ],
                "causal_claims_now": review.counters.get("causal_claims_generated", 0),
            }
        )
    shutil.rmtree(work, ignore_errors=True)
    document = {
        "run": args.name,
        "kind": "offline re-evaluation of stored summaries; no model was called",
        "limits": [
            "Operator checks were not stored by this run and are not re-evaluated.",
            "The evidence is reproduced by the deterministic pipeline; tool evidence the LLM "
            "requested in the original run may differ, which can only affect relevance checks.",
            "These figures are not part of the benchmark's counters.",
        ],
        "explanations": len(items),
        "accepted_in_run": sum(i["accepted_in_run"] for i in items),
        "summaries_accepted_now": sum(i["summary_accepted_now"] for i in items),
        "causal_claims_in_summaries_now": sum(i["causal_claims_now"] for i in items),
        "items": items,
    }
    path = out / "explanation_reevaluation.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(
        f"{document['explanations']} explanations: accepted in the run "
        f"{document['accepted_in_run']}, summaries accepted by the current validator "
        f"{document['summaries_accepted_now']} -> {path}"
    )
    for item in items:
        if not item["summary_accepted_now"]:
            print(
                f"  still rejected {item['dataset']} {item['fault_id']}: "
                f"{[c['text'][:110] for c in item['not_supported_claims_now']]}"
            )


if __name__ == "__main__":
    main()
