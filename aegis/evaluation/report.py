"""Aggregate AEGIS-IR raw results into metrics, tables, charts and a Markdown report.

    uv run python -m aegis.evaluation.report [--name main]

Reads ``results/aegis-ir/<name>/scenarios.jsonl`` and writes ``summary.json``,
``scenarios.csv``, ``report.md`` and ``charts/*.png`` next to it. Every number
in the report is computed here from the raw per-scenario records.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aegis.agents import claims as claim_rules
from aegis.evaluation.benchmark import SUPPORTED_FAULTS
from aegis.faults.injectors import FAULTS
from aegis.faults.scenarios import CLEAN
from aegis.remediation.reconciliation import benchmark_events, reconcile

TARGETS = {
    "drift_detection_f1": (0.90, ">="),
    "root_cause_recall_at_3": (0.80, ">="),
    "clean_control_false_positive_rate": (0.05, "<="),
    "unauthorized_tool_executions": (0, "=="),
    "invalid_promotions": (0, "=="),
    "supported_fault_remediation_success": (0.70, ">="),
}
#: Faults whose defining symptom is a covariate distribution change.
DRIFT_FAULTS = ("F01", "F02", "F04")
#: The two detection metrics. They differ in population AND in what counts as a detection, so
#: they are never interchangeable; every output names which one it is.
DETECTION_METRICS = {
    "overall_fault_detection": {
        "definition": (
            "Was an incident opened? Positives: every injected-fault scenario, all fault "
            "categories. Negatives: clean control batches. A scenario counts as detected when "
            "the monitoring run reports an incident, raised by any monitor."
        ),
        "positive_condition": "detection.incident_detected (any monitor)",
        "faults": None,  # all
    },
    "drift_detection": {
        "definition": (
            "Did the drift monitor flag a distribution change? Positives: only the covariate-"
            "drift scenarios (numerical shift F01, categorical shift F02, scaling error F04). "
            "Negatives: the same clean control batches. A scenario counts as detected only when "
            "the drift monitor itself triggers; an incident raised by another monitor does not "
            "count."
        ),
        "positive_condition": "detection.drift_flagged (the drift monitor specifically)",
        "faults": DRIFT_FAULTS,
    },
}
BOOTSTRAP_RESAMPLES = 2000
CONFIG_NAMES = {
    "A": "A  monitoring only",
    "B": "B  monitoring + rules",
    "C": "C  monitoring + local LLM",
    "D": "D  full AEGIS",
}
# Fixed categorical order (validated: adjacent CVD and normal-vision separation pass).
CONFIG_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
SEVERITY_COLORS = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}  # one hue, ordered


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well behaved for small n and p near 0/1)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


def rate(successes: int, n: int) -> dict:
    low, high = wilson(successes, n)
    return {
        "value": round(successes / n, 4) if n else None,
        "k": successes,
        "n": n,
        "ci95": [round(low, 3), round(high, 3)],
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def f1_bootstrap_ci(positive_hits: list[bool], negative_hits: list[bool]) -> list[float]:
    """95% percentile-bootstrap interval for F1, resampling scenarios within each class.

    F1 is not a proportion, so a Wilson interval does not apply. The seed is fixed: the same
    records always give the same interval.
    """
    if not positive_hits:
        return [0.0, 1.0]
    rng = np.random.default_rng(0)
    pos, neg = np.asarray(positive_hits, dtype=bool), np.asarray(negative_hits, dtype=bool)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))] if len(neg) else neg
        tp, fn, fp = int(p.sum()), int((~p).sum()), int(n.sum())
        values.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 3), round(float(high), 3)]


def detection_block(name: str, positives: list[dict], negatives: list[dict], flag) -> dict:
    """One detection metric with its population spelled out. ``name`` prefixes every value."""
    spec = DETECTION_METRICS[name]
    pos_hits, neg_hits = [bool(flag(r)) for r in positives], [bool(flag(r)) for r in negatives]
    tp, fn = sum(pos_hits), len(pos_hits) - sum(pos_hits)
    fp, tn = sum(neg_hits), len(neg_hits) - sum(neg_hits)
    scores = prf(tp, fp, fn)
    return {
        "metric_prefix": name,
        "definition": spec["definition"],
        "positive_condition": spec["positive_condition"],
        "population": {
            "included_fault_categories": sorted({r["truth"]["fault_id"] for r in positives}),
            "n_positive_scenarios": len(positives),
            "n_clean_scenarios": len(negatives),
            "n_scenarios": len(positives) + len(negatives),
        },
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        f"{name}_precision": {**rate(tp, tp + fp), "definition": "TP / (TP + FP)"},
        f"{name}_recall": {**rate(tp, tp + fn), "definition": "TP / (TP + FN)"},
        f"{name}_false_positive_rate": {**rate(fp, fp + tn), "definition": "FP / (FP + TN)"},
        f"{name}_f1": {
            "value": scores["f1"],
            "ci95": f1_bootstrap_ci(pos_hits, neg_hits),
            "ci_method": (
                f"percentile bootstrap over scenarios, {BOOTSTRAP_RESAMPLES} resamples, fixed seed"
            ),
            "definition": "2 TP / (2 TP + FP + FN)",
        },
    }


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def detection_metrics(records: list[dict]) -> dict:
    faulty = [r for r in records if r["truth"]["fault_id"] != CLEAN]
    clean = [r for r in records if r["truth"]["fault_id"] == CLEAN]
    hit = lambda r: r["detection"]["incident_detected"]  # noqa: E731
    out = {
        "overall_fault_detection": detection_block("overall_fault_detection", faulty, clean, hit)
    }
    for key in ("fault_id", "severity", "difficulty"):
        groups: dict[str, list] = defaultdict(list)
        for r in faulty:
            groups[r["truth"][key]].append(hit(r))
        out[f"recall_by_{key}"] = {k: rate(sum(v), len(v)) for k, v in sorted(groups.items())}
    out["recall_by_dataset"] = {
        d: rate(
            sum(hit(r) for r in faulty if r["dataset"] == d), sum(r["dataset"] == d for r in faulty)
        )
        for d in sorted({r["dataset"] for r in records})
    }
    out["false_positive_triggers"] = sorted(
        {t for r in clean if hit(r) for t in r["detection"]["triggered"]}
    )
    out["missed"] = [
        {
            "scenario_id": r["scenario_id"],
            "fault_id": r["truth"]["fault_id"],
            "severity": r["truth"]["severity"],
            "dataset": r["dataset"],
            "n_rows": r["n_rows"],
            "affected_features": r["truth"]["affected_features"],
        }
        for r in faulty
        if not hit(r)
    ]
    # Drift detection: a different population (covariate-drift faults only) and a different
    # positive condition (the drift monitor itself). Never to be read as the overall figure.
    drift_pos = [r for r in faulty if r["truth"]["fault_id"] in DRIFT_FAULTS]
    out["drift_detection"] = detection_block(
        "drift_detection", drift_pos, clean, lambda r: r["detection"]["drift_flagged"]
    )
    methods = {}
    for name in (
        "monitor",
        "ks",
        "ks_bh",
        "chi2",
        "chi2_bh",
        "psi_effect",
        "wasserstein_effect",
        "ks_effect",
    ):
        flag = (
            (lambda r: r["detection"]["drift_flagged"])
            if name == "monitor"
            else (lambda r, n=name: r["detection"]["drift_methods"].get(n, False))
        )
        m_tp, m_fn, m_fp = (
            sum(map(flag, drift_pos)),
            sum(not flag(r) for r in drift_pos),
            sum(map(flag, clean)),
        )
        methods[name] = {
            **prf(m_tp, m_fp, m_fn),
            "tn": len(clean) - m_fp,
            "false_positive_rate": round(m_fp / len(clean), 4) if clean else None,
        }
    out["drift_detection"]["methods_compared"] = {
        "note": "same population as drift_detection; 'monitor' is the shipped rule",
        "methods": methods,
    }
    times = [r["detection"]["monitoring_ms"] for r in records]
    out["monitoring_ms"] = {
        "p50": percentile(times, 50),
        "p95": percentile(times, 95),
        "max": round(max(times), 1),
    }
    return out


def diagnosis_metrics(
    records: list[dict], config: str, scope_ids: set[str] | None = None
) -> dict | None:
    faulty = [
        r
        for r in records
        if r["truth"]["fault_id"] != CLEAN and (scope_ids is None or r["scenario_id"] in scope_ids)
    ]
    ran = [r for r in faulty if config in r["configs"] and "error" not in r["configs"][config]]
    if not ran:
        return None
    rows = [r["configs"][config]["diagnosis"] for r in ran]
    crashed = sum(1 for r in faulty if "error" in r["configs"].get(config, {}))
    claims = sum(d["claims"] for d in rows)
    with_rival = [d for d in rows if len(d["ranked"]) > 1]
    by_fault: dict[str, list] = defaultdict(list)
    for r in ran:
        by_fault[r["truth"]["fault_id"]].append(r["configs"][config]["diagnosis"]["top1"])
    return {
        "scenarios_diagnosed": len(ran),
        "scenarios_in_scope": len(faulty),
        "crashed": crashed,
        "top1": rate(sum(d["top1"] for d in rows), len(rows)),
        "recall_at_3": rate(sum(d["recall_at_3"] for d in rows), len(rows)),
        # End-to-end: an undetected fault can never be diagnosed, so it counts as wrong.
        "top1_end_to_end": rate(sum(d["top1"] for d in rows), len(faulty)),
        "recall_at_3_end_to_end": rate(sum(d["recall_at_3"] for d in rows), len(faulty)),
        "evidence_coverage_mean": round(float(np.mean([d["evidence_coverage"] for d in rows])), 4),
        "contradictory_evidence_handling": rate(
            sum(d["contradictions_cited"] > 0 for d in with_rival), len(with_rival)
        ),
        "unsupported_claim_rate": rate(sum(d["unsupported_claims"] for d in rows), claims),
        "fabricated_evidence_ids": sum(d["cited_unknown_evidence"] for d in rows),
        "causal_overclaims": sum(d.get("overclaims", 0) for d in rows),
        "claim_validation": claim_metrics(rows),
        "insufficient_evidence_answers": sum(not d["sufficient_evidence"] for d in rows),
        "top1_by_fault": {k: rate(sum(v), len(v)) for k, v in sorted(by_fault.items())},
    }


def claim_metrics(rows: list[dict]) -> dict:
    """Aggregate the claim-validation counters recorded per investigation.

    Runs made before claim validation existed (``main``) carry no counters. For those the
    answer is "unavailable", never zero: zero would claim something was measured.
    """
    recorded = [d["claim_counters"] for d in rows if d.get("claim_counters")]
    if not recorded:
        return {
            "available": False,
            "reason": (
                "this run recorded no claim-validation counters (no LLM text was produced, or "
                "the run predates claim validation and did not store the model's text)"
            ),
        }
    total = claim_rules.add_counters(*recorded)
    for key in claim_rules.ESCAPE_COUNTERS:
        total.setdefault(key, 0)
    causal = total["causal_claims_generated"]
    fabricated = total["fabricated_evidence_ids_generated"]
    unsupported = total["unsupported_claims_generated"]
    return {
        "available": True,
        "investigations_with_llm_text": len(recorded),
        "counters": total,
        "rates": {
            "causal_claims_not_supported_rate": {
                **rate(total["causal_claims_rejected"] + total["causal_claims_escalated"], causal),
                "definition": "(causal claims rejected + escalated) / causal claims generated",
            },
            "causal_claims_escaped_rate": {
                **rate(total["causal_claims_escaped"], causal),
                "definition": (
                    "causal claims the validator still recognises in final output / causal "
                    "claims generated"
                ),
            },
            "unsupported_claims_escaped_rate": {
                **rate(total["unsupported_claims_escaped"], unsupported),
                "definition": (
                    "LLM-ranked hypotheses in final output with no anomaly evidence / "
                    "unsupported claims generated"
                ),
            },
            "fabricated_evidence_ids_escaped_rate": {
                **rate(total["fabricated_evidence_ids_escaped"], fabricated),
                "definition": (
                    "non-existent evidence ids in final output / fabricated ids generated"
                ),
            },
            "claims_with_causal_language_rate": {
                **rate(causal, total["claims_evaluated"]),
                "definition": "causal claims generated / all LLM claims evaluated",
            },
        },
        "limits": (
            "Generated counts are what a fixed lexicon recognises; an over-claim phrased "
            "outside it is not counted as generated and cannot be counted as escaped."
        ),
    }


def remediation_metrics(
    records: list[dict], config: str, scope_ids: set[str] | None = None
) -> dict | None:
    ran = [
        r
        for r in records
        if (scope_ids is None or r["scenario_id"] in scope_ids)
        and (r["configs"].get(config) or {}).get("remediation")
    ]
    if not ran:
        return None
    rem = lambda r: r["configs"][config]["remediation"]  # noqa: E731
    validated = [r for r in ran if rem(r)["validated"] is not None]
    executed = [r for r in ran if rem(r)["executed_resolves_incident"] is not None]
    rollbacks = [r for r in ran if rem(r)["rollback_restores_state"] is not None]
    supported = [
        r
        for r in records
        if r["truth"]["fault_id"] in SUPPORTED_FAULTS
        and r["detection"]["incident_detected"]
        and (scope_ids is None or r["scenario_id"] in scope_ids)
        and config in r["configs"]
    ]

    def supported_ok(r):
        x = (r["configs"][config] or {}).get("remediation")
        return bool(
            x and x["validated"] and x["action"] == SUPPORTED_FAULTS[r["truth"]["fault_id"]].value
        )

    actions: dict[str, int] = defaultdict(int)
    outcomes: dict[str, int] = defaultdict(int)
    for r in ran:
        actions[rem(r)["action"]] += 1
        outcomes[rem(r)["policy_outcome"]] += 1
    clean_actions = sum(
        1 for r in ran if r["truth"]["fault_id"] == CLEAN and rem(r)["approval_requested"]
    )
    scoped = [
        r
        for r in records
        if config in r["configs"] and (scope_ids is None or r["scenario_id"] in scope_ids)
    ]
    events, coverage = benchmark_events(scoped, config)
    lifecycle = reconcile(events, expected_proposals=coverage["investigated"], require_closed=True)
    no_action = sum(rem(r)["action"] in ("no_action", "investigate_further") for r in ran)

    def defined(item: dict, definition: str) -> dict:
        return {**item, "definition": definition}

    return {
        "counts": {
            "investigations": coverage["investigated"],
            "investigations_crashed": len(coverage["crashed"]),
            "investigations_without_remediation_record": len(coverage["without_remediation"]),
            "proposals_total": len(ran),
            "proposals_no_state_change": no_action,
            "proposals_actionable": len(ran) - no_action,
            "rejected_by_policy_before_validation": lifecycle["terminal_status_counts"].get(
                "rejected_by_policy_before_validation", 0
            ),
            "validation_attempts": len(validated),
            "validation_passed": sum(bool(rem(r)["validated"]) for r in validated),
            "validation_failed": sum(not rem(r)["validated"] for r in validated),
            "approval_requests": sum(bool(rem(r)["approval_requested"]) for r in ran),
            "executed_fixes": len(executed),
            "executed_fixes_resolving_incident": sum(
                bool(rem(r)["executed_resolves_incident"]) for r in executed
            ),
            "rollback_attempts": len(rollbacks),
            "rollbacks_succeeded": sum(bool(rem(r)["rollback_restores_state"]) for r in rollbacks),
            "supported_fault_scenarios_detected": len(supported),
        },
        "simulation_note": (
            "In the benchmark, approval, execution and rollback are simulated in-process (no "
            "human, API or database). The API/database path is exercised by the end-to-end tests."
        ),
        "proposals": len(ran),
        "actions": dict(sorted(actions.items())),
        "policy_outcomes": dict(sorted(outcomes.items())),
        "validation_success_rate": defined(
            rate(sum(bool(rem(r)["validated"]) for r in validated), len(validated)),
            "validation attempts that passed / all validation attempts",
        ),
        "regression_failure_rate": defined(
            rate(sum(bool(rem(r)["regression_failed"]) for r in validated), len(validated)),
            "validation attempts with a failed regression check / all validation attempts",
        ),
        "supported_fault_remediation_success": defined(
            rate(sum(map(supported_ok, supported)), len(supported)),
            "detected scenarios of a fault with an automatable fix where the expected action was "
            "proposed AND its validation passed / all detected scenarios of such faults. The "
            "denominator is scenarios, not validation attempts: a wrong or missing proposal "
            "counts as a failure.",
        ),
        "execution_resolution_rate": defined(
            rate(sum(bool(rem(r)["executed_resolves_incident"]) for r in executed), len(executed)),
            "executed fixes after which the incident batch was healthy / executed fixes",
        ),
        "rollback_success_rate": defined(
            rate(sum(bool(rem(r)["rollback_restores_state"]) for r in rollbacks), len(rollbacks)),
            "rollbacks that restored the previous state / rollback attempts",
        ),
        "invalid_promotions": sum(rem(r)["invalid_promotion"] for r in ran),
        "unauthorized_actions": sum(rem(r)["unauthorized_action"] for r in ran),
        "approval_requests_on_clean_data": clean_actions,
        "validation_ms_p50": percentile([rem(r)["validation_ms"] for r in validated], 50),
        "lifecycle": {
            "coverage": coverage,
            **{k: v for k, v in lifecycle.items() if k != "proposal_statuses"},
        },
    }


def agent_metrics(
    records: list[dict], config: str, scope_ids: set[str] | None = None
) -> dict | None:
    scoped = [
        r
        for r in records
        if config in r["configs"] and (scope_ids is None or r["scenario_id"] in scope_ids)
    ]
    ran = [r["configs"][config]["agent"] for r in scoped if "error" not in r["configs"][config]]
    if not ran:
        return None
    latency = [a["latency_ms"] for a in ran]
    llm_calls = sum(a.get("llm_calls", 0) for a in ran)
    out = {
        "investigations": len(scoped),
        "crashed": len(scoped) - len(ran),
        "completion_rate": rate(sum(a["completed"] for a in ran), len(scoped)),
        "tool_calls_mean": round(float(np.mean([a["tool_calls"] for a in ran])), 2),
        "tool_calls_max": int(max(a["tool_calls"] for a in ran)),
        "invalid_tool_arguments": sum(a.get("invalid_tool_args", 0) for a in ran),
        "unauthorized_tool_executions": sum(a.get("unauthorized_tool_executions", 0) for a in ran),
        "latency_ms": {"p50": percentile(latency, 50), "p95": percentile(latency, 95)},
    }
    if llm_calls:
        failures = sum(a.get("structured_output_failures", 0) for a in ran)
        out.update(
            {
                "llm_requests": llm_calls,
                "structured_output_failure_rate": rate(failures, llm_calls),
                "invalid_json_or_schema_attempts": sum(a.get("invalid_outputs", 0) for a in ran),
                "llm_timeouts_or_unavailable": sum(
                    a.get("llm_status") == "unavailable"
                    or a.get("llm_error") in ("timeout", "unavailable")
                    for a in ran
                ),
                "llm_ms_per_investigation": {
                    "p50": percentile([a["llm_ms"] for a in ran], 50),
                    "p95": percentile([a["llm_ms"] for a in ran], 95),
                },
                "llm_ms_per_request_mean": round(sum(a["llm_ms"] for a in ran) / llm_calls, 1),
                "prompt_tokens_mean": round(
                    float(np.mean([a.get("prompt_tokens") or 0 for a in ran])), 0
                ),
                "completion_tokens_mean": round(
                    float(np.mean([a.get("completion_tokens") or 0 for a in ran])), 0
                ),
            }
        )
    return out


def summarise(out_dir: Path) -> dict:
    records = [json.loads(line) for line in (out_dir / "scenarios.jsonl").read_text().splitlines()]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    subset = set(manifest.get("llm_subset") or [])
    llm_ran = {r["scenario_id"] for r in records if "C" in r["configs"] or "D" in r["configs"]}
    summary: dict = {
        "benchmark": "AEGIS-IR",
        "name": manifest["name"],
        "n_scenarios": len(records),
        "design": {
            k: manifest.get(k)
            for k in (
                "seeds",
                "dev_seeds_excluded",
                "clean_controls_per_dataset",
                "datasets",
                "faults",
                "model_versions",
                "llm_model",
                "llm_model_identity",
                "llm_options",
                "machine",
                "environment",
                "methodology",
            )
        },
        "detection": detection_metrics(records),
        "full_grid": {},
        "llm_subset": {"n": len(llm_ran), "configs": {}},
    }
    for config in ("A", "B"):
        summary["full_grid"][config] = {
            "diagnosis": diagnosis_metrics(records, config),
            "remediation": remediation_metrics(records, config),
            "agent": agent_metrics(records, config),
        }
    if llm_ran:
        for config in ("A", "B", "C", "D"):
            summary["llm_subset"]["configs"][config] = {
                "diagnosis": diagnosis_metrics(records, config, llm_ran),
                "remediation": remediation_metrics(records, config, llm_ran),
                "agent": agent_metrics(records, config, llm_ran),
            }
        summary["llm_subset"]["planned"] = len(subset)
    for name in ("delay", "resources"):
        path = out_dir / f"{name}.json"
        summary[name] = json.loads(path.read_text()) if path.exists() else None
    if summary["delay"]:
        rows = summary["delay"]
        found = [r["delay_windows"] for r in rows if r["delay_windows"] is not None]
        summary["delay_summary"] = {
            "streams": len(rows),
            "window_rows": rows[0]["window_rows"],
            "detected_in_first_window": rate(sum(d == 0 for d in found), len(rows)),
            "never_detected": sum(r["delay_windows"] is None for r in rows),
            "mean_delay_windows_when_detected": round(float(np.mean(found)), 3) if found else None,
            "false_alarm_rate_on_clean_prefix": rate(
                sum(r["false_alarms_before_onset"] for r in rows),
                sum(r["clean_windows"] for r in rows),
            ),
        }
    concept = [
        v for r in records for v in r["detection"]["concept_delay_rows"].values() if v is not None
    ]
    summary["concept_detector_delay_rows"] = {
        "detections": len(concept),
        "p50": percentile(concept, 50),
        "p95": percentile(concept, 95),
    }
    summary["headline_metrics"] = headline_metrics(summary)
    summary["targets"] = evaluate_targets(summary)
    return summary


def headline_metrics(summary: dict) -> dict:
    """Flat, explicitly named values. A name here is the only name a number may be quoted under."""
    out: dict = {}
    for name in DETECTION_METRICS:
        block = summary["detection"][name]
        for suffix in ("precision", "recall", "f1", "false_positive_rate"):
            out[f"{name}_{suffix}"] = block[f"{name}_{suffix}"]["value"]
        out[f"{name}_n_positive_scenarios"] = block["population"]["n_positive_scenarios"]
        out[f"{name}_n_clean_scenarios"] = block["population"]["n_clean_scenarios"]
    return out


def evaluate_targets(summary: dict) -> dict:
    best = (
        (summary["llm_subset"]["configs"].get("D") or {})
        if summary["llm_subset"]["configs"]
        else {}
    )
    full_b = summary["full_grid"]["B"]
    detection = summary["detection"]
    observed = {
        "drift_detection_f1": detection["drift_detection"]["drift_detection_f1"]["value"],
        "root_cause_recall_at_3": full_b["diagnosis"]["recall_at_3"]["value"],
        "clean_control_false_positive_rate": detection["overall_fault_detection"][
            "overall_fault_detection_false_positive_rate"
        ]["value"],
        "unauthorized_tool_executions": sum(
            (cfg.get("agent") or {}).get("unauthorized_tool_executions", 0)
            for cfg in [full_b, *summary["llm_subset"]["configs"].values()]
        ),
        "invalid_promotions": sum(
            (cfg.get("remediation") or {}).get("invalid_promotions", 0)
            for cfg in [full_b, *summary["llm_subset"]["configs"].values()]
        ),
        "supported_fault_remediation_success": full_b["remediation"][
            "supported_fault_remediation_success"
        ]["value"],
    }
    out = {}
    for key, (target, op) in TARGETS.items():
        value = observed[key]
        met = (
            value is not None
            and {">=": value >= target, "<=": value <= target, "==": value == target}[op]
        )
        out[key] = {"target": f"{op} {target}", "observed": value, "met": bool(met)}
    if best.get("diagnosis"):
        out["root_cause_recall_at_3"]["observed_full_aegis_llm_subset"] = best["diagnosis"][
            "recall_at_3"
        ]["value"]
    return out


# ----------------------------------------------------------------------------- output
def write_csv(out_dir: Path) -> None:
    records = [json.loads(line) for line in (out_dir / "scenarios.jsonl").read_text().splitlines()]
    fields = [
        "scenario_id",
        "dataset",
        "n_rows",
        "fault_id",
        "severity",
        "difficulty",
        "seed",
        "affected_features",
        "incident_detected",
        "max_severity",
        "monitoring_ms",
        "drift_flagged",
    ]
    per_config = [
        "top1",
        "recall_at_3",
        "ranked",
        "top_confidence",
        "action",
        "policy_outcome",
        "validated",
        "tool_calls",
        "latency_ms",
        "llm_ms",
    ]
    with (out_dir / "scenarios.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields + [f"{c}_{k}" for c in "ABCD" for k in per_config])
        for r in records:
            t, d = r["truth"], r["detection"]
            row = [
                r["scenario_id"],
                r["dataset"],
                r["n_rows"],
                t["fault_id"],
                t["severity"],
                t["difficulty"],
                t["seed"],
                "|".join(t["affected_features"]),
                d["incident_detected"],
                d["max_severity"],
                d["monitoring_ms"],
                d["drift_flagged"],
            ]
            for config in "ABCD":
                cfg = r["configs"].get(config)
                if not cfg or "error" in cfg:
                    row += [""] * len(per_config)
                    continue
                dg, rem, ag = cfg["diagnosis"], cfg.get("remediation") or {}, cfg["agent"]
                row += [
                    dg["top1"],
                    dg["recall_at_3"],
                    "|".join(dg["ranked"]),
                    dg["top_confidence"],
                    rem.get("action", ""),
                    rem.get("policy_outcome", ""),
                    rem.get("validated", ""),
                    ag.get("tool_calls", ""),
                    round(ag["latency_ms"], 1),
                    round(ag.get("llm_ms", 0) or 0, 1),
                ]
            writer.writerow(row)


def _style(ax, title: str, xlabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#52514e")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dedcd6")
    ax.tick_params(colors="#52514e", labelsize=9, length=0)
    ax.set_axisbelow(True)


def write_charts(summary: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "font.family": "DejaVu Sans",
        }
    )
    written = []

    # 1. Detection recall by fault category (one measure, one colour; values labelled).
    by_fault = summary["detection"]["recall_by_fault_id"]
    faults = sorted(by_fault)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    values = [by_fault[f]["value"] for f in faults]
    labels = [f"{f}  {FAULTS[f].name}" for f in faults]
    ax.barh(labels[::-1], values[::-1], height=0.55, color=CONFIG_COLORS["A"])
    for i, f in enumerate(faults[::-1]):
        ax.text(
            by_fault[f]["value"] + 0.012,
            i,
            f"{by_fault[f]['k']}/{by_fault[f]['n']}",
            va="center",
            fontsize=9,
            color="#0b0b0b",
        )
    ax.set_xlim(0, 1.12)
    ax.xaxis.grid(True, color="#ecebe7", linewidth=0.8)
    fpr = summary["detection"]["overall_fault_detection"][
        "overall_fault_detection_false_positive_rate"
    ]
    _style(
        ax,
        f"Detection recall by fault category  (clean-control false positives: {fpr['k']}/{fpr['n']})",
        "share of injected faults that opened an incident",
    )
    fig.tight_layout()
    fig.savefig(charts / "detection_recall_by_fault.png", dpi=150)
    plt.close(fig)
    written.append("detection_recall_by_fault.png")

    # 2. Diagnosis quality per configuration, on the scenarios every configuration saw.
    configs = summary["llm_subset"]["configs"] or {c: summary["full_grid"][c] for c in ("A", "B")}
    present = [c for c in "ABCD" if (configs.get(c) or {}).get("diagnosis")]
    scope = (
        f"LLM subset, n={summary['llm_subset']['n']} scenarios"
        if summary["llm_subset"]["configs"]
        else "full grid"
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("top1", "recall_at_3"),
        ("Root-cause Top-1 accuracy", "Root-cause Recall@3"),
        strict=True,
    ):
        for i, c in enumerate(present):
            item = configs[c]["diagnosis"][metric]
            low, high = item["ci95"]
            ax.bar(i, item["value"], width=0.5, color=CONFIG_COLORS[c], label=CONFIG_NAMES[c])
            ax.plot([i, i], [low, high], color="#52514e", linewidth=1.2)
            ax.text(
                i,
                min(high, 1.0) + 0.03,
                f"{item['value']:.2f}",
                ha="center",
                fontsize=9,
                color="#0b0b0b",
            )
        ax.set_xticks(
            range(len(present)),
            [CONFIG_NAMES[c].replace("  ", "\n", 1).replace(" + ", "\n+ ") for c in present],
            fontsize=8,
        )
        ax.set_ylim(0, 1.18)
        ax.yaxis.grid(True, color="#ecebe7", linewidth=0.8)
        _style(ax, title)
    fig.suptitle(
        f"Diagnosis by configuration ({scope}; whiskers: 95% Wilson interval)",
        x=0.01,
        ha="left",
        fontsize=10,
        color="#52514e",
    )
    fig.tight_layout()
    fig.savefig(charts / "diagnosis_by_config.png", dpi=150)
    plt.close(fig)
    written.append("diagnosis_by_config.png")

    # 3. Investigation latency per configuration (log axis: values span four orders of magnitude).
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, c in enumerate(present[::-1]):
        lat = configs[c]["agent"]["latency_ms"]
        p50, p95 = lat["p50"] / 1000, lat["p95"] / 1000
        # A dot, not a bar: on a log axis a bar's length depends on an arbitrary baseline.
        ax.plot([p50, p95], [i, i], color="#52514e", linewidth=2, solid_capstyle="round")
        ax.plot(
            [p50],
            [i],
            marker="o",
            markersize=9,
            color=CONFIG_COLORS[c],
            markeredgecolor="#fcfcfb",
            markeredgewidth=2,
        )
        ax.text(
            max(p95, p50) * 1.15,
            i,
            f"p50 {p50:.3g} s · p95 {p95:.3g} s",
            va="center",
            fontsize=9,
            color="#0b0b0b",
        )
    ax.set_yticks(range(len(present)), [CONFIG_NAMES[c] for c in present[::-1]])
    ax.set_xscale("log")
    ax.set_xlim(
        left=1e-4, right=max(configs[c]["agent"]["latency_ms"]["p95"] for c in present) / 1000 * 60
    )
    ax.xaxis.grid(True, color="#ecebe7", linewidth=0.8)
    _style(
        ax, "Investigation latency after detection (dot: p50, line to p95)", "seconds, log scale"
    )
    fig.tight_layout()
    fig.savefig(charts / "latency_by_config.png", dpi=150)
    plt.close(fig)
    written.append("latency_by_config.png")
    return written


def _pct(item: dict | None) -> str:
    if not item or item.get("value") is None:
        return "n/a"
    return f"{item['value']:.3f} ({item['k']}/{item['n']}, 95% CI {item['ci95'][0]:.2f}-{item['ci95'][1]:.2f})"


def _machine_line(design: dict) -> str:
    env = design.get("environment")
    if env:
        host = (env.get("host") or {}).get("physical_ram_gib")
        return (
            f"{env['physical_cores']} physical cores / {env['logical_cpus']} logical CPUs, "
            f"{env['guest_memory_total_gib']} GiB RAM visible to the OS"
            + (
                f" (WSL2 guest; host has {host} GiB installed)"
                if env.get("is_wsl") and host
                else ""
            )
        )
    machine = design.get("machine") or {}
    gib = round(machine.get("ram_gb", 0) * 1e9 / 2**30, 2)
    return (
        f"{machine.get('physical_cpus')} physical cores, {gib} GiB RAM visible to the OS "
        f"(recorded as {machine.get('ram_gb')} GB in decimal units; host RAM not recorded)"
    )


def _detection_table(block: dict) -> list[str]:
    name, pop, c = block["metric_prefix"], block["population"], block["confusion"]
    f1 = block[f"{name}_f1"]
    lines = [
        f"**`{name}`** - {block['definition']}",
        "",
        "| | |",
        "|---|---|",
        f"| Included fault categories | {', '.join(pop['included_fault_categories'])} |",
        f"| Positive scenarios | {pop['n_positive_scenarios']} |",
        f"| Clean (negative) scenarios | {pop['n_clean_scenarios']} |",
        f"| Detected when | `{block['positive_condition']}` |",
        f"| TP / FP / TN / FN | {c['tp']} / {c['fp']} / {c['tn']} / {c['fn']} |",
    ]
    for suffix in ("precision", "recall", "false_positive_rate"):
        item = block[f"{name}_{suffix}"]
        lines.append(f"| `{name}_{suffix}` = {item['definition']} | {_pct(item)} |")
    lines.append(
        f"| `{name}_f1` = {f1['definition']} | {f1['value']:.4f} "
        f"(95% CI {f1['ci95'][0]:.3f}-{f1['ci95'][1]:.3f}, {f1['ci_method']}) |"
    )
    return lines


def _resources_section(resources: dict | None) -> list[str]:
    if not resources:
        return []
    lines = ["", "## Resource consumption", ""]
    if "runs" not in resources:
        # Format written before the process-tree sampler existed (run `main`).
        lines += [
            "Historical measurements, taken with the original sampler. That sampler matched "
            "processes by name, so it could not see the Ollama runner (a child process named "
            "`llama-server`), and it recorded no swap, no sampling interval and no error count. "
            "Fields it did not or could not measure are shown as unavailable; they are not "
            "estimated from later runs.",
            "",
            "| Run | Duration (s) | Python peak RSS (MB) | Ollama RSS, one undivided figure (MB) | "
            "Ollama runner RSS (MB) | Process-tree total (MB) | System RAM used, peak (MB) | "
            "Swap | CPU mean / p95 (%) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for key, r in resources.items():
            ollama = (
                f"{r.get('ollama_peak_rss_mb')} (runner not captured)"
                if r.get("llm")
                else f"{r.get('ollama_peak_rss_mb')} (no model was loaded)"
            )
            lines.append(
                f"| {key} (historical) | {r.get('duration_s')} | {r.get('python_peak_rss_mb')} | "
                f"{ollama} | unavailable | unavailable | "
                f"{r.get('system_used_peak_mb')} | unavailable | "
                f"{r.get('cpu_percent_mean')} / {r.get('cpu_percent_p95')} |"
            )
        return lines
    lines += [
        "Each row is one separately measured invocation; rows are never merged. RSS values are "
        "sums of resident set sizes over a process group, each process counted once.",
        "",
        "| Run id | Label | Work done | Duration (s) | Interval (s) | Samples | Python (MB) | "
        "AEGIS tree (MB) | Ollama server (MB) | Ollama runner (MB) | Process-tree total (MB) | "
        "System used, peak (MB) | Swap used, peak (MB) | CPU mean / p95 (%) | Sampler errors |",
        "|" + "---|" * 15,
    ]
    for r in resources["runs"]:
        pk, cpu, work = r["peaks_mb"], r["cpu_percent"], r.get("work", {})
        done = ", ".join(f"{k}:{v}" for k, v in sorted(work.get("configs_executed", {}).items()))
        lines.append(
            f"| {r['run_id']} | {r['label']} | {done or 'nothing (already complete)'} | "
            f"{r['duration_s']} | {r['sampling_interval_s']} | {r['samples']} | "
            f"{pk['python_rss']} | {pk['aegis_tree_rss']} | {pk['ollama_server_rss']} | "
            f"{pk['ollama_runner_rss']} | {pk['process_tree_total_rss']} | {pk['system_used']} | "
            f"{pk['swap_used']} | {cpu['mean']} / {cpu['p95']} | {r['monitoring_error_total']} |"
        )
    for r in resources["runs"]:
        extra = [f"started {r['started_at']}, ended {r['ended_at']}"]
        if r.get("llm"):
            extra.append(
                f"model {r['llm'].get('name')} (digest {str(r['llm'].get('digest'))[:12]})"
            )
        extra += r.get("notes", [])
        if r.get("monitoring_errors"):
            extra.append(f"sampler errors: {r['monitoring_errors']}")
        lines.append(f"- `{r['run_id']}`: " + "; ".join(extra) + ".")
    env = resources.get("environment") or {}
    if env:
        host = env.get("host") or {}
        lines += [
            "",
            f"Environment: {env.get('cpu_model')}; {env.get('physical_cores')} physical cores, "
            f"{env.get('logical_cpus')} logical; memory visible to the OS "
            f"{env.get('guest_memory_total_gib')} GiB (= {env.get('guest_memory_total_gb_decimal')} GB "
            f"decimal); swap {env.get('guest_swap_total_gib')} GiB; "
            f"host installed RAM: {host.get('physical_ram_gib', 'unavailable')} GiB "
            f"({host.get('physical_ram_source', 'not queried')}); "
            f".wslconfig: {host.get('wslconfig', 'n/a')}.",
        ]
        lines += [f"- {text}" for text in env.get("limitations", [])]
    return lines


def write_markdown(summary: dict, out_dir: Path, charts: list[str]) -> None:
    det, design = summary["detection"], summary["design"]
    lines = [
        "# AEGIS-IR results",
        "",
        f"Run `{summary['name']}` · {summary['n_scenarios']} scenarios · datasets {design['datasets']} · "
        f"evaluation seeds {design['seeds']} (threshold-tuning seeds {design['dev_seeds_excluded']} excluded) · "
        f"machine: {_machine_line(design)}.",
        "",
        "Generated by `python -m aegis.evaluation.report` from `scenarios.jsonl`. Nothing here is typed by hand.",
    ]
    if design.get("methodology"):
        lines += ["", "Methodology notes for this run:", ""] + [
            f"- {note}" for note in design["methodology"]
        ]
    lines += [
        "",
        "## Engineering targets",
        "",
        "| Target (metric name) | Goal | Observed | Met |",
        "|---|---|---|---|",
    ]
    for key, item in summary["targets"].items():
        lines.append(
            f"| `{key}` | {item['target']} | {item['observed']} | {'yes' if item['met'] else '**no**'} |"
        )
    lines += [
        "",
        "## Detection (identical for all configurations)",
        "",
        "Two detection metrics are reported. They have different populations and different "
        "definitions of a detection, and must not be quoted for one another.",
        "",
        *_detection_table(det["overall_fault_detection"]),
        "",
        *_detection_table(det["drift_detection"]),
        "",
        f"Monitoring time per batch: p50 {det['monitoring_ms']['p50']} ms, p95 {det['monitoring_ms']['p95']} ms.",
        "",
        "### `overall_fault_detection_recall` by fault, severity and difficulty",
        "",
        "| Fault | Recall |",
        "|---|---|",
    ]
    lines += [f"| {f} {FAULTS[f].name} | {_pct(v)} |" for f, v in det["recall_by_fault_id"].items()]
    lines += ["", "| Severity | Recall |", "|---|---|"] + [
        f"| {k} | {_pct(v)} |" for k, v in det["recall_by_severity"].items()
    ]
    lines += ["", "| Difficulty | Recall |", "|---|---|"] + [
        f"| {k} | {_pct(v)} |" for k, v in det["recall_by_difficulty"].items()
    ]
    if det["missed"]:
        lines += ["", "Missed faults (no incident opened):", ""] + [
            f"- {m['fault_id']} {m['severity']} on {m['dataset']} (n={m['n_rows']}, features {m['affected_features']})"
            for m in det["missed"]
        ]
    if det["false_positive_triggers"]:
        lines += ["", f"Checks that fired on clean controls: {det['false_positive_triggers']}"]
    drift = det["drift_detection"]
    lines += [
        "",
        "### Drift methods compared (population of `drift_detection`)",
        "",
        f"Positives: the {drift['population']['n_positive_scenarios']} F01/F02/F04 scenarios. "
        f"Negatives: the {drift['population']['n_clean_scenarios']} clean controls. `monitor` is the "
        "shipped rule (BH-adjusted significance AND a material effect size) and is the row "
        "reported as `drift_detection_*` above.",
        "",
        "| Method | Precision | Recall | F1 | TP / FP / TN / FN | Clean FPR |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
        f"{m['tp']} / {m['fp']} / {m['tn']} / {m['fn']} | {m['false_positive_rate']} |"
        for name, m in drift["methods_compared"]["methods"].items()
    ]
    if summary.get("delay_summary"):
        ds = summary["delay_summary"]
        lines += [
            "",
            "### Detection delay",
            "",
            f"Streams of clean windows followed by faulty windows ({ds['window_rows']} rows each, {ds['streams']} streams): "
            f"detected in the first faulty window {_pct(ds['detected_in_first_window'])}; never detected {ds['never_detected']}; "
            f"mean delay when detected {ds['mean_delay_windows_when_detected']} windows; "
            f"false alarms on the clean prefix {_pct(ds['false_alarm_rate_on_clean_prefix'])}.",
            f"ADWIN / Page-Hinkley delay on the labelled error stream: p50 {summary['concept_detector_delay_rows']['p50']} rows, "
            f"p95 {summary['concept_detector_delay_rows']['p95']} rows ({summary['concept_detector_delay_rows']['detections']} detections).",
        ]

    def table(scope: dict, title: str, note: str) -> None:
        present = [c for c in "ABCD" if (scope.get(c) or {}).get("diagnosis")]
        if not present:
            return
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                note,
                "",
                "| Metric | " + " | ".join(present) + " |",
                "|---|" + "---|" * len(present),
            ]
        )

        def row(label, getter):
            cells = []
            for c in present:
                try:
                    value = getter(scope[c])
                except (KeyError, TypeError):
                    value = "n/a"
                cells.append("n/a" if value is None else str(value))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        row("Scenarios diagnosed", lambda s: s["diagnosis"]["scenarios_diagnosed"])
        row("Root-cause Top-1 (correct / diagnosed)", lambda s: _pct(s["diagnosis"]["top1"]))
        row(
            "Root-cause Recall@3 (correct / diagnosed)",
            lambda s: _pct(s["diagnosis"]["recall_at_3"]),
        )
        row(
            "Top-1, end-to-end (correct / all faulty in scope; a missed detection is wrong)",
            lambda s: _pct(s["diagnosis"]["top1_end_to_end"]),
        )
        row("Evidence coverage (mean)", lambda s: s["diagnosis"]["evidence_coverage_mean"])
        row(
            "Contradictory evidence cited (diagnoses citing it / diagnoses with a rival)",
            lambda s: _pct(s["diagnosis"]["contradictory_evidence_handling"]),
        )
        row(
            "Validation success (passed / validation attempts)",
            lambda s: _pct(s["remediation"]["validation_success_rate"]),
        )
        row(
            "Supported-fault remediation success (expected action validated / detected supported-fault scenarios)",
            lambda s: _pct(s["remediation"]["supported_fault_remediation_success"]),
        )
        row(
            "Regression failures (failed regression check / validation attempts)",
            lambda s: _pct(s["remediation"]["regression_failure_rate"]),
        )
        row(
            "Execution resolution (incident cleared / executed fixes)",
            lambda s: _pct(s["remediation"]["execution_resolution_rate"]),
        )
        row(
            "Rollback success (state restored / rollback attempts)",
            lambda s: _pct(s["remediation"]["rollback_success_rate"]),
        )
        row("Invalid promotions", lambda s: s["remediation"]["invalid_promotions"])
        row("Unauthorized actions", lambda s: s["remediation"]["unauthorized_actions"])
        row("Unauthorized tool executions", lambda s: s["agent"]["unauthorized_tool_executions"])
        row(
            "Diagnostic tool calls (mean / max)",
            lambda s: f"{s['agent']['tool_calls_mean']} / {s['agent']['tool_calls_max']}",
        )
        row(
            "Invalid tool arguments (LLM requests rejected)",
            lambda s: s["agent"]["invalid_tool_arguments"],
        )
        row(
            "Structured-output failures (failed / LLM requests)",
            lambda s: _pct(s["agent"]["structured_output_failure_rate"]),
        )
        row("LLM timeouts / unavailable", lambda s: s["agent"]["llm_timeouts_or_unavailable"])
        row(
            "Investigation completion (completed / investigations)",
            lambda s: _pct(s["agent"]["completion_rate"]),
        )
        row(
            "Investigation latency p50 / p95 (s)",
            lambda s: (
                f"{s['agent']['latency_ms']['p50'] / 1000:.3g} / {s['agent']['latency_ms']['p95'] / 1000:.3g}"
            ),
        )
        row(
            "LLM time per request, mean (s)",
            lambda s: round(s["agent"]["llm_ms_per_request_mean"] / 1000, 1),
        )

    def claims_table(scope: dict, title: str) -> None:
        present = [c for c in "CD" if (scope.get(c) or {}).get("diagnosis")]
        if not present:
            return
        lines.extend(["", f"### Claim validation: {title}", ""])
        blocks = {c: scope[c]["diagnosis"]["claim_validation"] for c in present}
        available = [c for c in present if blocks[c]["available"]]
        for c in present:
            if not blocks[c]["available"]:
                legacy = scope[c]["diagnosis"]
                lines.append(
                    f"- Configuration {c}: **unavailable** - {blocks[c]['reason']}. "
                    f"Legacy counters recorded at the time: ranked items dropped as unsupported or "
                    f"unknown {legacy['unsupported_claim_rate']['k']} of {legacy['unsupported_claim_rate']['n']}; "
                    f"fabricated evidence ids stripped {legacy['fabricated_evidence_ids']}; matches of the "
                    f"original over-claim pattern on ranking reasoning {legacy['causal_overclaims']}. "
                    "The last figure says only that the old pattern did not match; it does not "
                    "show that no causal claim was generated, and nothing was measured about "
                    "escapes."
                )
        if not available:
            return
        lines.extend(
            [
                "",
                "| Counter | " + " | ".join(available) + " |",
                "|---|" + "---|" * len(available),
            ]
        )
        keys = [*claim_rules.COUNTERS, *claim_rules.ESCAPE_COUNTERS]
        for key in keys:
            lines.append(
                f"| `{key}` | "
                + " | ".join(str(blocks[c]["counters"].get(key, 0)) for c in available)
                + " |"
            )
        lines.extend(
            [
                "",
                "| Rate (numerator / denominator) | " + " | ".join(available) + " |",
                "|---|" + "---|" * len(available),
            ]
        )
        for key in blocks[available[0]]["rates"]:
            definition = blocks[available[0]]["rates"][key]["definition"]
            lines.append(
                f"| `{key}` = {definition} | "
                + " | ".join(_pct(blocks[c]["rates"][key]) for c in available)
                + " |"
            )
        lines.extend(["", blocks[available[0]]["limits"]])

    def lifecycle_section(rem: dict, title: str) -> None:
        counts, lc = rem["counts"], rem["lifecycle"]
        lines.extend(
            [
                "",
                f"### Remediation lifecycle accounting: {title}",
                "",
                rem["simulation_note"],
                "",
                "| Stage | Count | How it follows from the stage above |",
                "|---|---|---|",
                f"| Investigations (detected faulty scenarios) | {counts['investigations']} | |",
                f"| Proposals recorded | {counts['proposals_total']} | one per investigation; "
                f"{counts['investigations_crashed']} crashed, "
                f"{counts['investigations_without_remediation_record']} without a record |",
                f"| - no state change proposed (`no_action` / `investigate_further`) | {counts['proposals_no_state_change']} | policy `DEFER` / `INVESTIGATE_MORE`; never validated |",
                f"| - actionable proposals | {counts['proposals_actionable']} | {counts['proposals_total']} - {counts['proposals_no_state_change']} |",
                f"| -- rejected by policy before validation | {counts['rejected_by_policy_before_validation']} | policy `REJECT` on its first pass (reasons are in the lifecycle events when recorded); never validated |",
                f"| -- validation attempts | {counts['validation_attempts']} | {counts['proposals_actionable']} - {counts['rejected_by_policy_before_validation']}"
                + (
                    f" - {lc['terminal_status_counts'].get('investigate_more', 0)} investigate-more"
                    if lc["terminal_status_counts"].get("investigate_more")
                    else ""
                )
                + " |",
                f"| --- validation failed (policy `REJECT`) | {counts['validation_failed']} | |",
                f"| --- validation passed -> approval requested | {counts['approval_requests']} | {counts['validation_attempts']} - {counts['validation_failed']} |",
                f"| ---- executed (simulated approval) | {counts['executed_fixes']} | every approval request is approved in the benchmark |",
                f"| ----- incident resolved after execution | {counts['executed_fixes_resolving_incident']} | |",
                f"| ----- rollback attempts / succeeded | {counts['rollback_attempts']} / {counts['rollbacks_succeeded']} | every execution is rolled back in the benchmark |",
                "",
                f"Reconciliation: **{'reconciled' if lc['reconciled'] else 'FAILED'}** - "
                f"{lc['events']} lifecycle events for {lc['proposals']} proposals "
                f"({lc['coverage']['recorded']} recorded live, "
                f"{lc['coverage']['derived_from_legacy_fields']} derived from legacy summary fields; "
                f"{lc['simulated_events']} events simulated). Terminal statuses: {lc['terminal_status_counts']}.",
            ]
        )
        if lc["failures"]:
            lines.extend(["", "Reconciliation failures:", ""])
            lines.extend(
                f"- `{f['code']}` {f.get('proposal_id') or ''}: {f['message']}"
                for f in lc["failures"][:40]
            )
        lines.extend(["", "| Accounting identity | Left | Right | Holds |", "|---|---|---|---|"])
        lines.extend(
            f"| {i['name']} | {i['left']} | {i['right']} | {'yes' if i['holds'] else '**no**'} |"
            for i in lc["accounting_identities"]
        )
        lines.extend(
            [
                "",
                f"Supported-fault remediation success uses a different denominator from the rows "
                f"above: {counts['supported_fault_scenarios_detected']} detected scenarios whose "
                "fault has an automatable fix (F03, F04, F06, F07, F08, F10). "
                f"{rem['supported_fault_remediation_success']['definition']}",
                "",
                f"Actions proposed: {rem['actions']}",
                "",
                f"Final policy outcomes: {rem['policy_outcomes']}",
                "",
                f"Approval requests raised on clean data: {rem['approval_requests_on_clean_data']}.",
            ]
        )

    table(
        summary["full_grid"],
        "Deterministic configurations on the full grid",
        "A = alerts only; B = diagnostics + rules + policy + sandbox validation. Diagnosis is scored on detected faults.",
    )
    if summary["full_grid"]["B"].get("remediation"):
        lifecycle_section(summary["full_grid"]["B"]["remediation"], "configuration B, full grid")
    if summary["llm_subset"]["configs"]:
        table(
            summary["llm_subset"]["configs"],
            f"All four configurations on the LLM subset (n={summary['llm_subset']['n']})",
            "One scenario per dataset x fault category, severity rotated. Same scenarios for A, B, C and D. "
            "With n this small the intervals are wide; read differences accordingly.",
        )
        claims_table(summary["llm_subset"]["configs"], "LLM subset")
        if (summary["llm_subset"]["configs"].get("D") or {}).get("remediation"):
            lifecycle_section(
                summary["llm_subset"]["configs"]["D"]["remediation"], "configuration D, LLM subset"
            )
    lines += _resources_section(summary.get("resources"))
    lines += ["", "## Charts", ""] + [f"![{c}](charts/{c})" for c in charts] + [""]
    (out_dir / "report.md").write_text("\n".join(lines))


def write_lifecycle(out_dir: Path) -> dict:
    """Full machine-readable lifecycle reconciliation, one entry per configuration."""
    records = [json.loads(line) for line in (out_dir / "scenarios.jsonl").read_text().splitlines()]
    output: dict = {"run": out_dir.name, "configurations": {}}
    for config in ("B", "D"):
        events, coverage = benchmark_events(records, config)
        if not coverage["investigated"]:
            continue
        report = reconcile(events, expected_proposals=coverage["investigated"], require_closed=True)
        output["configurations"][config] = {
            "coverage": coverage,
            **report,
            "events_detail": [e.model_dump(mode="json") for e in events],
        }
    output["reconciled"] = all(c["reconciled"] for c in output["configurations"].values())
    (out_dir / "lifecycle_reconciliation.json").write_text(json.dumps(output, indent=1) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="main")
    parser.add_argument("--out", type=Path, default=Path("results/aegis-ir"))
    args = parser.parse_args()
    out_dir = args.out / args.name
    summary = summarise(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out_dir)
    lifecycle = write_lifecycle(out_dir)
    charts = write_charts(summary, out_dir)
    write_markdown(summary, out_dir, charts)
    print(json.dumps(summary["targets"], indent=2))
    print(json.dumps(summary["headline_metrics"], indent=2))
    print(f"lifecycle reconciled: {lifecycle['reconciled']}")
    print(f"report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
