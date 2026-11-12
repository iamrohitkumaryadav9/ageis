"""Benchmark metric integrity: populations, names and denominators cannot be mixed silently."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aegis.evaluation import report as R

MAIN = Path("results/aegis-ir/main")
FAULTS = [f"F{i:02d}" for i in range(1, 11)]


def record(fault, *, detected, drift_flagged, sid=None):
    return {
        "scenario_id": sid or f"SCN-{fault}-{detected}-{drift_flagged}-{id(object())}",
        "dataset": "adult",
        "n_rows": 1000,
        "truth": {
            "fault_id": fault,
            "severity": "medium",
            "difficulty": "medium",
            "affected_features": [],
        },
        "detection": {
            "incident_detected": detected,
            "drift_flagged": drift_flagged,
            "drift_methods": {},
            "triggered": [],
            "monitoring_ms": 100.0,
            "concept_delay_rows": {},
        },
        "configs": {},
    }


def synthetic():
    """A grid where the two metrics *must* disagree.

    Every fault is detected as an incident, but the drift monitor fires on only half of the
    drift faults and on one clean control that opened no incident.
    """
    rows = []
    for fault in FAULTS:
        for k in range(4):
            drift = fault in R.DRIFT_FAULTS and k < 2
            rows.append(record(fault, detected=True, drift_flagged=drift))
    rows += [record("CLEAN", detected=False, drift_flagged=(k == 0)) for k in range(10)]
    return rows


@pytest.fixture(scope="module")
def det():
    return R.detection_metrics(synthetic())


def test_the_two_metrics_use_different_populations(det):
    overall, drift = det["overall_fault_detection"], det["drift_detection"]
    assert overall["population"]["n_positive_scenarios"] == 40
    assert overall["population"]["included_fault_categories"] == FAULTS
    assert drift["population"]["n_positive_scenarios"] == 12
    assert drift["population"]["included_fault_categories"] == ["F01", "F02", "F04"]
    assert (
        overall["population"]["n_clean_scenarios"] == drift["population"]["n_clean_scenarios"] == 10
    )


def test_the_two_metrics_use_different_positive_conditions(det):
    overall, drift = det["overall_fault_detection"], det["drift_detection"]
    assert overall["confusion"] == {"tp": 40, "fp": 0, "tn": 10, "fn": 0}
    assert drift["confusion"] == {"tp": 6, "fp": 1, "tn": 9, "fn": 6}
    assert overall["overall_fault_detection_f1"]["value"] == 1.0
    assert drift["drift_detection_f1"]["value"] == round(2 * 6 / (2 * 6 + 1 + 6), 4)
    assert "any monitor" in overall["positive_condition"]
    assert "drift monitor" in drift["positive_condition"]


def test_confusion_cells_cover_the_population_exactly(det):
    for name in R.DETECTION_METRICS:
        block = det[name]
        c, pop = block["confusion"], block["population"]
        assert c["tp"] + c["fn"] == pop["n_positive_scenarios"]
        assert c["fp"] + c["tn"] == pop["n_clean_scenarios"]
        assert sum(c.values()) == pop["n_scenarios"]


def test_every_value_is_named_with_its_metric_prefix(det):
    for name in R.DETECTION_METRICS:
        block = det[name]
        other = next(n for n in R.DETECTION_METRICS if n != name)
        scored = [k for k in block if k.endswith(("_f1", "_precision", "_recall", "_rate"))]
        assert sorted(scored) == sorted(
            f"{name}_{s}" for s in ("precision", "recall", "false_positive_rate", "f1")
        )
        assert not any(k.startswith(other) for k in block)
        for key in scored:
            assert block[key]["definition"] and "ci95" in block[key]
    assert "f1" not in det and "overall" not in det and "drift" not in det  # no unnamed scores


def test_rates_carry_numerators_denominators_and_intervals(det):
    recall = det["drift_detection"]["drift_detection_recall"]
    assert (recall["k"], recall["n"], recall["value"]) == (6, 12, 0.5)
    low, high = recall["ci95"]
    assert 0 <= low < 0.5 < high <= 1
    fpr = det["drift_detection"]["drift_detection_false_positive_rate"]
    assert (fpr["k"], fpr["n"]) == (1, 10)


def test_f1_interval_is_deterministic_and_contains_the_estimate(det):
    again = R.detection_metrics(synthetic())
    for name in R.DETECTION_METRICS:
        f1 = det[name][f"{name}_f1"]
        assert f1 == again[name][f"{name}_f1"]
        assert f1["ci95"][0] <= f1["value"] <= f1["ci95"][1]
        assert "bootstrap" in f1["ci_method"]


def test_precision_with_no_positive_predictions_is_none_not_zero_or_one():
    rows = [record("F01", detected=False, drift_flagged=False) for _ in range(3)]
    rows += [record("CLEAN", detected=False, drift_flagged=False) for _ in range(3)]
    block = R.detection_metrics(rows)["overall_fault_detection"]
    assert block["overall_fault_detection_precision"]["value"] is None
    assert block["overall_fault_detection_recall"]["value"] == 0.0


def test_moving_a_scenario_between_populations_changes_only_the_right_metric():
    base = R.detection_metrics(synthetic())
    rows = synthetic()
    # one more non-drift fault that is missed: overall recall falls, drift metrics do not move
    rows.append(record("F06", detected=False, drift_flagged=False))
    changed = R.detection_metrics(rows)
    assert changed["drift_detection"] == base["drift_detection"]
    assert (
        changed["overall_fault_detection"]["overall_fault_detection_recall"]["value"]
        < base["overall_fault_detection"]["overall_fault_detection_recall"]["value"]
    )


def test_drift_method_comparison_stays_inside_the_drift_population(det):
    methods = det["drift_detection"]["methods_compared"]["methods"]
    monitor = methods["monitor"]
    assert monitor["tp"] + monitor["fn"] == 12 and monitor["fp"] + monitor["tn"] == 10
    assert monitor["f1"] == det["drift_detection"]["drift_detection_f1"]["value"]


# ------------------------------------------------------------ historical run
@pytest.fixture(scope="module")
def main_summary():
    return R.summarise(MAIN)


def test_historical_numbers_are_reproduced_from_the_raw_file(main_summary):
    h = main_summary["headline_metrics"]
    assert h["overall_fault_detection_f1"] == 0.9958
    assert h["overall_fault_detection_precision"] == 1.0
    assert h["overall_fault_detection_recall"] == 0.9917
    assert (
        h["overall_fault_detection_n_positive_scenarios"],
        h["overall_fault_detection_n_clean_scenarios"],
    ) == (120, 50)
    assert h["drift_detection_f1"] == 0.9565
    assert (h["drift_detection_n_positive_scenarios"], h["drift_detection_n_clean_scenarios"]) == (
        36,
        50,
    )
    overall = main_summary["detection"]["overall_fault_detection"]["confusion"]
    drift = main_summary["detection"]["drift_detection"]["confusion"]
    assert overall == {"tp": 119, "fp": 0, "tn": 50, "fn": 1}
    assert drift == {"tp": 33, "fp": 0, "tn": 50, "fn": 3}


def test_raw_historical_file_is_unmodified():
    import hashlib

    digest = hashlib.sha256((MAIN / "scenarios.jsonl").read_bytes()).hexdigest()
    assert digest == "910170c1f1119b6d119b7b361767fcf4b0c8d32d43d3ccde94767c9d715374f5"
    archived = MAIN / "history" / "scenarios.2026-11-11.before-noise-allowance.jsonl"
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == (
        "702f79384cc53fc27637248c6815b26a3fbfb592fe2faccc5908754d1d017d75"
    )


def test_targets_use_explicit_metric_names(main_summary):
    targets = main_summary["targets"]
    assert "drift_f1" not in targets and "clean_fpr" not in targets and "recall_at_3" not in targets
    assert targets["drift_detection_f1"]["observed"] == 0.9565
    assert set(targets) == set(R.TARGETS)


def test_remediation_rates_state_their_denominators(main_summary):
    rem = main_summary["full_grid"]["B"]["remediation"]
    counts = rem["counts"]
    assert (rem["validation_success_rate"]["k"], rem["validation_success_rate"]["n"]) == (68, 85)
    assert rem["validation_success_rate"]["n"] == counts["validation_attempts"]
    supported = rem["supported_fault_remediation_success"]
    assert (supported["k"], supported["n"]) == (67, 72)
    assert (
        supported["n"]
        == counts["supported_fault_scenarios_detected"]
        != counts["validation_attempts"]
    )
    assert "scenarios, not validation attempts" in supported["definition"]
    assert rem["execution_resolution_rate"]["n"] == counts["executed_fixes"] == 68
    assert rem["rollback_success_rate"]["n"] == counts["rollback_attempts"] == 68
    assert counts["proposals_actionable"] == 87 and counts["validation_attempts"] == 85
    assert (
        counts["proposals_actionable"] - counts["validation_attempts"]
        == counts["rejected_by_policy_before_validation"]
        == 2
    )
    for key in (
        "validation_success_rate",
        "regression_failure_rate",
        "execution_resolution_rate",
        "rollback_success_rate",
        "supported_fault_remediation_success",
    ):
        assert rem[key]["definition"]
    assert rem["lifecycle"]["reconciled"] is True


def test_claim_metrics_of_a_run_without_counters_are_unavailable_not_zero(main_summary):
    for config in ("C", "D"):
        block = main_summary["llm_subset"]["configs"][config]["diagnosis"]["claim_validation"]
        assert block["available"] is False and "counters" not in block


def test_claim_metrics_aggregate_and_define_every_rate():
    rows = [
        {
            "claim_counters": {
                "claims_evaluated": 4,
                "causal_claims_generated": 2,
                "causal_claims_rejected": 1,
                "causal_claims_escalated": 1,
                "fabricated_evidence_ids_generated": 3,
                "fabricated_evidence_ids_stripped": 3,
            }
        },
        {
            "claim_counters": {
                "claims_evaluated": 2,
                "causal_claims_generated": 1,
                "causal_claims_supported": 1,
                "causal_claims_escaped": 0,
            }
        },
        {},  # an investigation with no LLM text
    ]
    block = R.claim_metrics(rows)
    assert block["available"] and block["investigations_with_llm_text"] == 2
    c = block["counters"]
    assert c["causal_claims_generated"] == 3
    assert c["causal_claims_generated"] == (
        c["causal_claims_supported"] + c["causal_claims_rejected"] + c["causal_claims_escalated"]
    )
    not_supported = block["rates"]["causal_claims_not_supported_rate"]
    assert (not_supported["k"], not_supported["n"]) == (2, 3)
    assert block["rates"]["fabricated_evidence_ids_escaped_rate"]["n"] == 3
    assert all(r["definition"] for r in block["rates"].values())
    assert set(R.claim_rules.ESCAPE_COUNTERS) <= set(c)


# -------------------------------------------------------------- report text
@pytest.fixture(scope="module")
def main_report(main_summary, tmp_path_factory):
    out = tmp_path_factory.mktemp("report")
    R.write_markdown(main_summary, out, charts=[])
    return (out / "report.md").read_text()


def test_report_never_prints_an_unnamed_f1(main_report):
    for line in main_report.splitlines():
        if re.search(r"\bF1\b|_f1\b", line) and re.search(r"0\.9\d", line) and line.startswith("|"):
            named = "overall_fault_detection_f1" in line or "drift_detection_f1" in line
            method_row = (
                line.split("|")[1].strip()
                in R.detection_metrics(synthetic())["drift_detection"]["methods_compared"][
                    "methods"
                ]
            )
            assert named or method_row, line


def test_report_keeps_each_value_with_its_own_population(main_report):
    overall = next(x for x in main_report.splitlines() if "`overall_fault_detection_f1`" in x)
    drift = next(x for x in main_report.splitlines() if "`drift_detection_f1`" in x and "=" in x)
    assert "0.9958" in overall and "0.9565" not in overall
    assert "0.9565" in drift and "0.9958" not in drift
    assert "Positive scenarios | 120" in main_report and "Positive scenarios | 36" in main_report
    assert "must not be quoted for one another" in main_report


DOCS = [Path("README.md"), Path("docs/benchmark.md"), Path("docs/interview-preparation.md")]
_OVERALL = re.compile(
    r"overall|ten (injected )?(ML )?(failure|fault)|all (ten )?fault|fault-detection", re.I
)
_DRIFT = re.compile(r"drift", re.I)


@pytest.mark.parametrize("path", DOCS, ids=str)
def test_documents_never_attach_a_detection_f1_to_the_wrong_population(path):
    """0.9565/0.957 belongs to drift detection, 0.996/0.9958 to overall fault detection."""
    text = path.read_text()
    for raw in text.splitlines():
        # a sentence may legitimately contain both numbers; check each clause around a number
        for match in re.finditer(r"0\.95(?:65|7)\b|0\.99(?:6|58)\b", raw):
            number = match.group()
            window = raw[max(0, match.start() - 110) : match.end() + 40]
            if number.startswith("0.95"):
                assert _DRIFT.search(window), f"{path}: {number} without 'drift' nearby: {raw}"
            elif re.search(r"\bF1\b|f1", window):
                assert _OVERALL.search(window), f"{path}: {number} F1 without 'overall': {raw}"
                before = raw[max(0, match.start() - 40) : match.start()]
                assert not re.search(r"drift[- ]detection F1\s*$", before, re.I), raw


def test_readme_headline_sentence_matches_the_recomputed_values(main_summary):
    readme = Path("README.md").read_text()
    h = main_summary["headline_metrics"]
    sentence = (
        f"{h['overall_fault_detection_f1']:.3f} overall fault-detection F1 across ten injected ML "
        f"failure categories and {h['drift_detection_f1']} drift-detection F1 across numerical, "
        "categorical, and scaling drift scenarios"
    )
    assert sentence in " ".join(readme.split()), "README headline is missing or out of date"
