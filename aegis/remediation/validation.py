"""Sandbox validation of a remediation proposal.

"Isolated" means: validation works on in-memory copies of the batch and on a
*candidate* deployment state. It never writes the live deployment state. The
only side effect allowed is registering a retrained candidate model in the
(append-only) registry, where it is inert until a deployment points at it.

Every action is checked for four things:
  precondition  the situation the action is meant for really holds
  effect        the targeted anomaly clears, measured on held-out data where possible
  regression    nothing that was fine gets worse (no new findings, performance holds)
  contract      the data the model will see conforms to the data contract
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from aegis.diagnosis.features import inspect_feature
from aegis.diagnosis.performance import evaluate_version
from aegis.ingestion.datasets import TARGET
from aegis.models.metrics import best_f1_threshold
from aegis.models.training import build_pipeline, candidate_estimators, predict_proba
from aegis.monitoring.config import MonitoringConfig
from aegis.monitoring.engine import MonitoringInput, MonitoringResult, run_monitoring
from aegis.monitoring.relationship import hanley_mcneil_se
from aegis.remediation.proposals import candidate_state
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.findings import Severity
from aegis.schemas.remediation import (
    ActionType,
    RemediationProposal,
    ValidationCheck,
    ValidationResult,
)

A = ActionType
PERFORMANCE_TOLERANCE = 0.03  # allowed AUC shortfall versus the reference
RETRAIN_MIN_GAIN = 0.02
RETRAIN_MAX_REFERENCE_REGRESSION = 0.10
MIN_HOLDOUT = 200


def _check(name, passed, detail, category="effect") -> ValidationCheck:
    return ValidationCheck(name=name, passed=bool(passed), detail=detail, category=category)


def _alerts(result: MonitoringResult) -> set[tuple[str, str | None]]:
    return {
        (f.check, f.feature)
        for f in result.report.triggered
        if f.severity.rank >= Severity.MEDIUM.rank
    }


def _auc(result: MonitoringResult) -> float | None:
    current = result.performance.get("current")
    return current.get("roc_auc") if current else None


def _perf_checks(before: MonitoringResult, after: MonitoringResult) -> list[ValidationCheck]:
    reference = after.performance.get("reference") or {}
    ref_auc, after_auc, before_auc = reference.get("roc_auc"), _auc(after), _auc(before)
    if after_auc is None or ref_auc is None:
        return [
            _check(
                "performance_not_degraded",
                True,
                "no labelled rows available; performance could not be compared (not a pass on merit)",
                "regression",
            )
        ]
    # "Not worse than before" needs a noise allowance: two AUCs measured on the same few
    # thousand rows differ by sampling error alone. Two Hanley-McNeil standard errors is the
    # allowance; without it a correct fix is rejected for an AUC dip well inside the noise.
    current = after.performance.get("current") or {}
    n = int(current.get("n") or 0)
    n_pos = round(n * float(current.get("positive_rate_true") or 0.0))
    noise = 2 * hanley_mcneil_se(after_auc, n_pos, n - n_pos) if 0 < n_pos < n else 0.0
    before_floor = before_auc - noise if before_auc is not None else 1.0
    floor = min(ref_auc - PERFORMANCE_TOLERANCE, before_floor)
    return [
        _check(
            "performance_not_degraded",
            after_auc >= floor - 1e-9,
            f"ROC-AUC after {after_auc:.3f} vs before "
            f"{'n/a' if before_auc is None else f'{before_auc:.3f}'}, reference {ref_auc:.3f} "
            f"(floor {floor:.3f}; sampling-noise allowance {noise:.3f})",
            "regression",
        )
    ]


def _common(
    before: MonitoringResult, after: MonitoringResult, cleared: set
) -> list[ValidationCheck]:
    new = _alerts(after) - _alerts(before)
    remaining = _alerts(after) & cleared
    schema_bad = [a for a in _alerts(after) if a[0].startswith("schema_")]
    return [
        _check(
            "targeted_findings_cleared",
            not remaining,
            "all targeted findings cleared"
            if not remaining
            else f"still triggered: {sorted(remaining)}",
        ),
        _check(
            "no_new_findings",
            not new,
            "no new findings" if not new else f"new findings appeared: {sorted(new, key=str)}",
            "regression",
        ),
        _check(
            "data_contract_valid",
            not schema_bad,
            "corrected data conforms to the contract"
            if not schema_bad
            else f"violations: {schema_bad}",
            "contract",
        ),
        *_perf_checks(before, after),
    ]


def validate_proposal(
    proposal: RemediationProposal,
    inp: MonitoringInput,
    result: MonitoringResult,
    state: DeploymentState,
    cfg: MonitoringConfig | None = None,
    seed: int = 0,
) -> ValidationResult:
    started = time.perf_counter()
    try:
        outcome = _VALIDATORS[proposal.action](proposal, inp, result, state, cfg, seed)
    except KeyError:
        outcome = ValidationResult(
            passed=False,
            checks=[
                _check(
                    "action_supported",
                    False,
                    f"{proposal.action.value} has nothing to validate",
                    "precondition",
                )
            ],
        )
    except Exception as exc:  # a crashing validation is a failed validation, never a pass
        outcome = ValidationResult(
            passed=False,
            checks=[
                _check(
                    "validation_completed",
                    False,
                    f"validation raised {type(exc).__name__}: {exc}",
                    "precondition",
                )
            ],
        )
    outcome.passed = bool(outcome.checks) and all(c.passed for c in outcome.checks)
    failed = [c.name for c in outcome.checks if not c.passed]
    outcome.summary = "all checks passed" if outcome.passed else f"failed: {failed}"
    outcome.duration_ms = round((time.perf_counter() - started) * 1000, 1)
    if outcome.passed and outcome.candidate_state is None:
        outcome.candidate_state = candidate_state(state, proposal).model_dump()
    if not outcome.passed:
        outcome.candidate_state = None
    return outcome


def _halves(inp: MonitoringInput):
    frame = inp.current.reset_index(drop=True)
    return frame.iloc[::2].reset_index(drop=True), frame.iloc[1::2].reset_index(drop=True)


def _fix_scale(proposal, inp, result, state, cfg, seed) -> ValidationResult:
    feature, factor = proposal.parameters.get("feature"), proposal.parameters.get("factor")
    kinds = {c.name: c.kind for c in inp.contract.columns}
    pre = _check(
        "parameters_valid",
        kinds.get(feature) == "numeric"
        and isinstance(factor, int | float)
        and np.isfinite(factor)
        and factor > 0
        and abs(np.log(factor)) > 0.05,
        f"feature={feature!r}, factor={factor!r}",
        "precondition",
    )
    if not pre.passed:
        return ValidationResult(passed=False, checks=[pre])
    fit_half, holdout = _halves(inp)
    # The factor must be reproducible from one half and must work on the other.
    estimate = inspect_feature(
        feature, "numeric", inp.reference, fit_half, inp.spec.numeric_features
    )
    stable = (
        estimate.pattern == "scale"
        and estimate.scale_factor
        and abs(np.log(estimate.scale_factor * factor)) < np.log(1.15)
    )
    new_state = candidate_state(state, proposal)
    before = run_monitoring(replace(inp, current=holdout, ingestion=state.ingestion), cfg)
    after = run_monitoring(replace(inp, current=holdout, ingestion=new_state.ingestion), cfg)
    checks = [
        pre,
        _check(
            "factor_reproducible_on_independent_half",
            stable,
            f"half-sample estimate x{estimate.scale_factor} ({estimate.pattern}) vs proposed inverse {factor:.4g}",
        ),
        *_common(before, after, {("numeric_drift", feature), ("range_violation_rate", feature)}),
    ]
    return ValidationResult(
        passed=False,
        checks=checks,
        metrics_before=before.performance.get("current") or {},
        metrics_after=after.performance.get("current") or {},
    )


def _dedup(proposal, inp, result, state, cfg, seed) -> ValidationResult:
    new_state = candidate_state(state, proposal)
    before = run_monitoring(replace(inp, ingestion=state.ingestion), cfg)
    after = run_monitoring(replace(inp, ingestion=new_state.ingestion), cfg)
    removed = before.report.n_current - after.report.n_current
    checks = [
        _check(
            "duplicates_present",
            ("duplicate_rate", None) in _alerts(before),
            "duplicate-rate finding is active before the fix",
            "precondition",
        ),
        _check("rows_removed", removed > 0, f"{removed} duplicate rows removed"),
        *_common(before, after, {("duplicate_rate", None)}),
    ]
    return ValidationResult(
        passed=False,
        checks=checks,
        metrics_before=before.performance.get("current") or {},
        metrics_after=after.performance.get("current") or {},
    )


def _restore(proposal, inp, result, state, cfg, seed) -> ValidationResult:
    version = proposal.parameters.get("version", "")
    try:
        record = inp.registry.get(inp.spec.name, version)
        ok, actual = inp.registry.verify(inp.spec.name, version)
    except (KeyError, ValueError):
        return ValidationResult(
            passed=False,
            checks=[
                _check(
                    "target_version_registered",
                    False,
                    f"{version!r} is not a registered version",
                    "integrity",
                )
            ],
        )
    checks = [
        _check(
            "target_artifact_hash_verified",
            ok,
            f"registry artifact sha256 {actual[:12]}... {'matches' if ok else 'DOES NOT match'} the record",
            "integrity",
        )
    ]
    if not ok:
        return ValidationResult(passed=False, checks=checks)
    on_reference = evaluate_version(inp.registry, inp.spec, version, inp.reference)
    recorded = (record.metrics.get("validation") or {}).get("roc_auc")
    if on_reference and recorded is not None:
        checks.append(
            _check(
                "restored_model_reproduces_recorded_performance",
                abs(on_reference["roc_auc"] - recorded) <= 0.01,
                f"ROC-AUC on reference {on_reference['roc_auc']:.4f} vs recorded {recorded:.4f}",
                "regression",
            )
        )
    elif on_reference:
        checks.append(
            _check(
                "restored_model_scores_reference",
                on_reference["roc_auc"] > 0.5,
                f"ROC-AUC on reference {on_reference['roc_auc']:.4f}",
                "regression",
            )
        )
    before = result
    after = run_monitoring(
        replace(inp, model_version=version, artifact_path=None, ingestion=state.ingestion), cfg
    )
    checks += _common(before, after, {("artifact_hash", None)})
    return ValidationResult(
        passed=False,
        checks=checks,
        metrics_before=before.performance.get("current") or {},
        metrics_after=after.performance.get("current") or {},
    )


def _quarantine(proposal, inp, result, state, cfg, seed) -> ValidationResult:
    grounds = {
        "schema_missing_columns",
        "schema_type_mismatch",
        "missing_rate",
        "label_association_gain",
    }
    active = sorted({check for check, _ in _alerts(result)} & grounds)
    record = inp.registry.get(inp.spec.name, state.model_version)
    ok, _ = inp.registry.verify(inp.spec.name, state.model_version)
    new_state = candidate_state(state, proposal)
    checks = [
        _check(
            "grounds_for_quarantine_present",
            bool(active),
            f"active findings justifying quarantine: {active or 'none'}",
            "precondition",
        ),
        _check(
            "model_and_ingestion_unchanged",
            new_state.model_version == state.model_version
            and new_state.ingestion == state.ingestion,
            "quarantine does not alter the model or the ingestion config",
            "regression",
        ),
        _check(
            "deployed_model_intact",
            ok,
            f"registry artifact for {record.version} verifies",
            "integrity",
        ),
    ]
    return ValidationResult(passed=False, checks=checks)


def _retrain(proposal, inp, result, state, cfg, seed) -> ValidationResult:
    spec = inp.spec
    current = result.current
    usable = set(spec.features) <= set(current.columns) and TARGET in current.columns
    labelled = current[current[TARGET].notna()] if usable else current.iloc[0:0]
    pre = _check(
        "enough_labelled_recent_data",
        len(labelled) >= 2 * MIN_HOLDOUT and usable and labelled[TARGET].nunique() == 2,
        f"{len(labelled)} labelled, contract-conforming rows (need >= {2 * MIN_HOLDOUT})",
        "precondition",
    )
    if not pre.passed:
        return ValidationResult(passed=False, checks=[pre])
    y = labelled[TARGET].astype(int)
    fit, holdout = train_test_split(labelled, test_size=0.4, stratify=y, random_state=seed)
    deployed = inp.registry.get(spec.name, state.model_version)
    estimator, params = candidate_estimators(seed)[deployed.algorithm]
    candidate = build_pipeline(spec, estimator)
    candidate.fit(fit[spec.features], fit[TARGET].astype(int))

    def auc(model, frame):
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(frame[TARGET].astype(int), predict_proba(model, frame, spec.features))
        )

    live = inp.registry.load(spec.name, state.model_version)
    cand_holdout, live_holdout = auc(candidate, holdout), auc(live, holdout)
    cand_ref, live_ref = auc(candidate, inp.reference), auc(live, inp.reference)
    checks = [
        pre,
        _check(
            "candidate_beats_deployed_on_heldout_recent_data",
            cand_holdout - live_holdout >= RETRAIN_MIN_GAIN,
            f"held-out recent ROC-AUC: candidate {cand_holdout:.3f} vs deployed {live_holdout:.3f} "
            f"(need +{RETRAIN_MIN_GAIN}) on {len(holdout)} rows",
        ),
        _check(
            "regression_on_reference_within_bound",
            live_ref - cand_ref <= RETRAIN_MAX_REFERENCE_REGRESSION,
            f"reference ROC-AUC: candidate {cand_ref:.3f} vs deployed {live_ref:.3f} "
            f"(max allowed drop {RETRAIN_MAX_REFERENCE_REGRESSION})",
            "regression",
        ),
    ]
    out = ValidationResult(
        passed=False,
        checks=checks,
        metrics_before={"holdout_roc_auc": live_holdout, "reference_roc_auc": live_ref},
        metrics_after={"holdout_roc_auc": cand_holdout, "reference_roc_auc": cand_ref},
    )
    if all(c.passed for c in checks):
        threshold = best_f1_threshold(
            fit[TARGET].astype(int).to_numpy(), predict_proba(candidate, fit, spec.features)
        )
        record = inp.registry.register(
            {
                "dataset": spec.name,
                "version": f"{spec.name}-cand-{result.report.report_id.removeprefix('REP-')}",
                "algorithm": deployed.algorithm,
                "threshold": threshold,
                "features": spec.features,
                "contract_version": inp.contract.version,
                "train_fingerprint": "recent-batch",
                "seed": seed,
                "params": params,
                "parent_version": deployed.version,
                "metrics": {
                    "validation": {"roc_auc": cand_ref},
                    "holdout_recent": {"roc_auc": cand_holdout},
                },
                "library_versions": {"scikit-learn": sklearn.__version__},
                "notes": "retrained candidate from sandbox validation; not deployed until approved",
            },
            candidate,
        )
        out.candidate_model_version = record.version
        promoted = proposal.model_copy(
            update={"parameters": {**proposal.parameters, "version": record.version}}
        )
        out.candidate_state = candidate_state(state, promoted).model_dump()
    return out


_VALIDATORS = {
    A.FIX_INGESTION_SCALE: _fix_scale,
    A.ENABLE_DEDUPLICATION: _dedup,
    A.RESTORE_MODEL_VERSION: _restore,
    A.QUARANTINE_BATCH: _quarantine,
    A.RETRAIN_CANDIDATE: _retrain,
}
