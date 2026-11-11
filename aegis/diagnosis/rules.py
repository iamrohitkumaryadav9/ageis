"""Rule-based hypothesis ranking over evidence records.

Each rule scores one candidate root cause from the evidence that supports it
and the evidence that argues against it. The score is an *evidence-fit*
value in [0, 1] used for ranking; it is not a probability that the candidate
is the true cause, and the output never says "caused by".

The rules encode the discriminating observations between causes that produce
overlapping symptoms:

* shift vs. scale    -> additive or multiplicative quantile relationship
* drift vs. leakage  -> leakage changes the feature/label link, not the marginal
* concept vs. drift  -> errors rise while inputs look normal
* anything vs. wrong model / broken schema / missing labels -> observed facts
  that are checked first because they invalidate downstream statistics
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from aegis.schemas.diagnosis import (
    INSUFFICIENT_EVIDENCE,
    Diagnosis,
    Evidence,
    Hypothesis,
    RootCause,
)

MIN_CONFIDENCE = 0.5
PERF_CHECKS = ("accuracy_drop", "precision_drop", "recall_drop", "f1_drop", "roc_auc_drop")
QUALITY_CHECKS = (
    "missing_rate",
    "range_violation_rate",
    "schema_missing_columns",
    "schema_type_mismatch",
    "duplicate_rate",
)


class EvidenceIndex:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.all = evidence

    def anomalies(self, *checks: str) -> list[Evidence]:
        return [
            e
            for e in self.all
            if e.data.get("check") in checks and e.data.get("polarity") == "anomaly"
        ]

    def normal(self, name: str) -> list[Evidence]:
        return [e for e in self.all if e.data.get("check") == f"no_{name}"]

    def patterns(self) -> dict[str, Evidence]:
        return {
            e.feature: e for e in self.all if e.data.get("check") == "feature_pattern" and e.feature
        }


def _ids(items: list[Evidence]) -> list[str]:
    return list(dict.fromkeys(e.evidence_id for e in items))


def _hyp(cause, confidence, support, contra, reasoning, *, features=(), missing=(), next_action=""):
    support_ids, contra_ids = _ids(support), _ids(contra)
    digest = hashlib.sha256((cause.value + "|".join(sorted(support_ids))).encode()).hexdigest()[:8]
    return Hypothesis(
        hypothesis_id=f"HYP-{digest}",
        root_cause=cause,
        affected_features=sorted(dict.fromkeys(features)),
        supporting_evidence_ids=support_ids,
        contradicting_evidence_ids=contra_ids,
        confidence=round(min(max(confidence, 0.0), 0.99), 3),
        reasoning=reasoning,
        missing_evidence=list(missing),
        next_diagnostic_action=next_action,
    )


def _artifact(ix: EvidenceIndex) -> Hypothesis | None:
    hits = ix.anomalies("artifact_hash")
    if not hits:
        return None
    known = hits[0].data.get("matches_registered_version")
    return _hyp(
        RootCause.ARTIFACT_MISMATCH,
        0.97,
        hits + ix.anomalies("prediction_distribution"),
        [],
        "The serving artifact's hash differs from its registry record (an observed fact). "
        + (
            f"The bytes equal registered version {known}, consistent with the wrong version being deployed."
            if known
            else "The bytes match no registered version, so the file was not loaded or scored."
        ),
        missing=[] if known else ["what the unregistered artifact is and how it reached serving"],
        next_action="compare_model_versions"
        if known
        else "restore the registered artifact, then re-monitor",
    )


def _schema(ix: EvidenceIndex) -> Hypothesis | None:
    hits = ix.anomalies("schema_missing_columns", "schema_type_mismatch")
    if not hits:
        return None
    return _hyp(
        RootCause.SCHEMA_CHANGE,
        0.95,
        hits + ix.anomalies("schema_unexpected_columns"),
        [],
        "The batch violates the data contract (columns absent or re-typed), which is an observed "
        "fact; the model cannot score it, so downstream statistics are unavailable rather than normal.",
        features=[e.feature for e in hits if e.feature],
        missing=["which upstream change altered the schema"],
        next_action="get_data_quality_report",
    )


def _label_delay(ix: EvidenceIndex) -> Hypothesis | None:
    hits = ix.anomalies("label_availability")
    if not hits:
        return None
    return _hyp(
        RootCause.LABEL_DELAY,
        0.9,
        hits,
        [],
        "A large share of rows has no label yet. Supervised metrics are unavailable or computed "
        "on a partial, possibly unrepresentative subset; this says nothing about model quality.",
        missing=["whether labels are late or lost"],
        next_action="get_model_metrics",
    )


def _duplicates(ix: EvidenceIndex) -> Hypothesis | None:
    hits = ix.anomalies("duplicate_rate")
    if not hits:
        return None
    excess = float(hits[0].data.get("excess") or 0.0)
    return _hyp(
        RootCause.DUPLICATE_RECORDS,
        min(0.92, 0.72 + excess),
        hits,
        [],
        "The exact-duplicate row rate is well above the size-matched reference expectation, "
        "consistent with replayed or double-ingested records.",
        missing=["whether duplicates share an ingestion batch or key"],
        next_action="get_data_quality_report",
    )


def _missing(ix: EvidenceIndex) -> Hypothesis | None:
    hits = ix.anomalies("missing_rate")
    if not hits:
        return None
    excess = max(float(e.data.get("excess") or 0.0) for e in hits)
    return _hyp(
        RootCause.MISSING_VALUES,
        min(0.92, 0.72 + excess),
        hits,
        [],
        "Missingness rose sharply for specific features while their non-null values are "
        "distributed as before, consistent with an upstream field no longer being populated.",
        features=[e.feature for e in hits if e.feature],
        missing=["whether missingness correlates with a source or time window"],
        next_action="inspect_feature_distribution",
    )


def _numeric(ix: EvidenceIndex) -> list[Hypothesis]:
    drifted = ix.anomalies("numeric_drift")
    ranged = {e.feature: e for e in ix.anomalies("range_violation_rate")}
    patterns = ix.patterns()
    if not drifted and not ranged:
        return []
    scale_fit, scale_sup, scale_con, scale_feats = 0.0, [], [], []
    shift_fit, shift_sup, shift_con, shift_feats = 0.0, [], [], []
    uninspected = []
    for e in drifted:
        pattern_ev = patterns.get(e.feature)
        pattern = pattern_ev.data.get("pattern") if pattern_ev else None
        in_range = ranged.get(e.feature)
        if pattern == "scale":
            scale_fit = max(scale_fit, 0.9)
            scale_sup += [e, pattern_ev] + ([in_range] if in_range else [])
            scale_feats.append(e.feature)
            shift_fit = max(shift_fit, 0.15)
            shift_sup.append(e)
            shift_con.append(pattern_ev)
        elif pattern == "shift":
            shift_fit = max(shift_fit, 0.88)
            shift_sup += [e, pattern_ev]
            shift_feats.append(e.feature)
            scale_fit = max(scale_fit, 0.15)
            scale_sup.append(e)
            scale_con.append(pattern_ev)
        elif pattern in ("ambiguous", "shape_change"):
            for fit_name in ("scale", "shift"):
                value = 0.5 if pattern == "ambiguous" else (0.35 if fit_name == "scale" else 0.6)
                if fit_name == "scale":
                    scale_fit, scale_sup = max(scale_fit, value), [*scale_sup, e, pattern_ev]
                    scale_feats.append(e.feature)
                else:
                    shift_fit, shift_sup = max(shift_fit, value), [*shift_sup, e, pattern_ev]
                    shift_feats.append(e.feature)
        else:  # drift seen, pattern not inspected yet
            uninspected.append(e.feature)
            shift_fit, shift_sup = max(shift_fit, 0.55), [*shift_sup, e]
            shift_feats.append(e.feature)
            value = 0.6 if in_range else 0.3
            scale_fit = max(scale_fit, value)
            scale_sup += [e] + ([in_range] if in_range else [])
            scale_feats.append(e.feature)
    for feature, e in ranged.items():  # out-of-range values with no drift finding
        if feature in {d.feature for d in drifted}:
            continue
        pattern_ev = patterns.get(feature)
        pattern = pattern_ev.data.get("pattern") if pattern_ev else None
        if pattern == "scale":  # e.g. a sparse feature: few non-zero values, all multiplied
            scale_fit, scale_sup = max(scale_fit, 0.88), [*scale_sup, e, pattern_ev]
        elif pattern == "shift":
            shift_fit, shift_sup = max(shift_fit, 0.7), [*shift_sup, e, pattern_ev]
            shift_feats.append(feature)
            scale_fit, scale_sup, scale_con = (
                max(scale_fit, 0.2),
                [*scale_sup, e],
                [*scale_con, pattern_ev],
            )
        else:
            scale_fit, scale_sup = max(scale_fit, 0.5), [*scale_sup, e]
            if pattern is None:
                uninspected.append(feature)
        scale_feats.append(feature)
    missing = [f"additive-vs-multiplicative pattern for {uninspected}"] if uninspected else []
    out = []
    if scale_fit:
        out.append(
            _hyp(
                RootCause.SCALING_ERROR,
                scale_fit,
                scale_sup,
                scale_con,
                "A numeric feature's quantiles relate to the reference by a constant factor, the "
                "signature of a unit or scaling change rather than a change in the population."
                if scale_fit >= 0.8
                else "A numeric feature moved; a multiplicative (unit/scale) change is one candidate but the "
                "quantile pattern does not single it out.",
                features=scale_feats,
                missing=[*missing, "the upstream unit or transformation that changed"],
                next_action="inspect_feature_distribution",
            )
        )
    if shift_fit:
        out.append(
            _hyp(
                RootCause.NUMERICAL_SHIFT,
                shift_fit,
                shift_sup,
                shift_con,
                "A numeric feature's quantiles moved by a near-constant offset, consistent with a "
                "location shift in the incoming population."
                if shift_fit >= 0.8
                else "A numeric feature's distribution moved; a location shift is one candidate.",
                features=shift_feats,
                missing=[*missing, "whether the population or the measurement changed"],
                next_action="inspect_feature_distribution",
            )
        )
    return out


def _categorical(ix: EvidenceIndex) -> Hypothesis | None:
    drifted = ix.anomalies("categorical_drift")
    unknown = ix.anomalies("unexpected_category_rate")
    if not drifted and not unknown:
        return None
    fit = (
        0.6
        if not drifted
        else (0.85 if any((e.data.get("observed") or 0) >= 0.25 for e in drifted) else 0.8)
    )
    return _hyp(
        RootCause.CATEGORICAL_SHIFT,
        fit,
        drifted + unknown + ix.anomalies("cardinality_change"),
        [],
        "Category proportions of a feature differ materially from the reference.",
        features=[e.feature for e in drifted + unknown if e.feature],
        missing=["whether the mix of the population changed or an encoding/mapping changed"],
        next_action="inspect_feature_distribution",
    )


def _leakage(ix: EvidenceIndex) -> Hypothesis | None:
    hits = ix.anomalies("label_association_gain")
    if not hits:
        return None
    drifted = {e.feature: e for e in ix.anomalies("numeric_drift", "categorical_drift")}
    fit = 0.75 + (0.15 if any((e.data.get("observed") or 0) >= 0.8 for e in hits) else 0.0)
    contra = [drifted[e.feature] for e in hits if e.feature in drifted]
    fit += -0.1 if contra else 0.05
    return _hyp(
        RootCause.FEATURE_LEAKAGE,
        fit,
        hits + (ix.normal("covariate_drift") if not contra else []),
        contra,
        "A single feature's association with the label rose far above its reference level"
        + (" while its own distribution is unchanged" if not contra else "")
        + ". Features do not normally become more predictive on their own; this pattern is "
        "consistent with label information leaking into the feature.",
        features=[e.feature for e in hits if e.feature],
        missing=["lineage of the feature: is it computed after the outcome is known?"],
        next_action="inspect_feature_distribution",
    )


def _concept(ix: EvidenceIndex) -> Hypothesis | None:
    drops = ix.anomalies(*PERF_CHECKS)
    alarms = ix.anomalies("adwin", "page_hinkley")
    reversal = ix.anomalies("label_association_reversal")
    if not drops and not alarms and not reversal:
        return None
    fit = 0.7 if drops and alarms else 0.5
    fit += 0.1 if reversal else 0.0
    support = drops + alarms + reversal
    contra: list[Evidence] = []
    inputs_normal = ix.normal("covariate_drift") + ix.normal("quality_violation")
    drift = ix.anomalies("numeric_drift", "categorical_drift")
    quality = ix.anomalies(*QUALITY_CHECKS)
    if drift:
        fit -= 0.3
        contra += drift
    if quality:
        fit -= 0.3
        contra += quality
    if ix.anomalies("artifact_hash"):
        fit -= 0.5
        contra += ix.anomalies("artifact_hash")
    if ix.anomalies("label_association_gain"):
        fit -= 0.35
        contra += ix.anomalies("label_association_gain")
    if ix.anomalies("label_availability"):
        fit -= 0.1
        contra += ix.anomalies("label_availability")
    if not drift and not quality and len(inputs_normal) == 2:
        fit += 0.15
        support = support + inputs_normal
    reasoning = (
        "Model errors rose while the inputs look like the reference, which is consistent with "
        "a change in the relationship between features and the label."
        if not contra
        else "Model errors rose, but other anomalies are present that could explain the loss without "
        "any change in the feature/label relationship."
    )
    return _hyp(
        RootCause.CONCEPT_DRIFT,
        max(fit, 0.05),
        support,
        contra,
        reasoning,
        missing=[
            "label quality for the period",
            "whether the change is global or confined to a segment",
        ],
        next_action="get_model_metrics",
    )


RULES: list[Callable[[EvidenceIndex], Hypothesis | list[Hypothesis] | None]] = [
    _artifact,
    _schema,
    _label_delay,
    _duplicates,
    _missing,
    _numeric,
    _categorical,
    _leakage,
    _concept,
]


def rank_hypotheses(evidence: list[Evidence]) -> Diagnosis:
    index = EvidenceIndex(evidence)
    hypotheses: list[Hypothesis] = []
    for rule in RULES:
        produced = rule(index)
        if produced is None:
            continue
        hypotheses += produced if isinstance(produced, list) else [produced]
    hypotheses.sort(key=lambda h: (-h.confidence, h.root_cause.value))
    return Diagnosis(
        hypotheses=hypotheses,
        engine="rules",
        **conclude(hypotheses),
    )


def conclude(hypotheses: list[Hypothesis]) -> dict:
    """Shared conclusion logic: below the threshold the honest answer is 'insufficient'."""
    if not hypotheses or hypotheses[0].confidence < MIN_CONFIDENCE:
        return {"conclusion": INSUFFICIENT_EVIDENCE, "sufficient_evidence": False}
    top = hypotheses[0]
    text = (
        f"The evidence is most consistent with {top.root_cause.value.replace('_', ' ')} "
        f"(evidence fit {top.confidence:.2f})"
        + (f" affecting {top.affected_features}" if top.affected_features else "")
        + ". This is a ranked hypothesis, not established causation."
    )
    if len(hypotheses) > 1 and top.confidence - hypotheses[1].confidence < 0.1:
        text += (
            f" {hypotheses[1].root_cause.value.replace('_', ' ')} fits nearly as well "
            f"({hypotheses[1].confidence:.2f}) and is not ruled out."
        )
    return {"conclusion": text, "sufficient_evidence": True}
