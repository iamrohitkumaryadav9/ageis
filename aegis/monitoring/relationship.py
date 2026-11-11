"""Feature/label relationship monitoring (needs labels).

For each feature the strength of its *univariate* association with the label
is compared between reference and current data. A sharp increase is the
signature of leakage (a feature that suddenly encodes the outcome); a sign
flip is a signature of a changed concept. Monotone transformations of a
feature (shift, rescale) leave this statistic unchanged, which is what makes
it useful for telling leakage apart from covariate drift.

Strength is ``|2*AUC - 1|``. A gain is flagged only if it is both material and
several standard errors large (Hanley-McNeil AUC variance). Categorical
strength uses in-sample target encoding, which is optimistic for small samples,
so its reference value is computed on reference subsamples of the same size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aegis.ingestion.contracts import DataContract
from aegis.monitoring.common import make_finding
from aegis.monitoring.config import MonitoringConfig
from aegis.schemas.findings import EpistemicLevel, Finding, Severity


def signed_association(feature: pd.Series, y: np.ndarray, kind: str) -> tuple[float, float] | None:
    """Return (2*AUC-1, standard error of that quantity), or None if not computable.

    Numeric: the raw feature is the score. Categorical: the per-category positive
    rate is the score (so the sign is always >= 0 and only magnitude is meaningful).
    """
    mask = feature.notna().to_numpy()
    if mask.sum() < 50 or np.unique(y[mask]).size < 2:
        return None
    values, labels = feature[mask], y[mask]
    if kind == "numeric":
        score = values.to_numpy(dtype=float)
    else:
        as_text = values.astype(str)
        rates = pd.Series(labels, index=as_text.index).groupby(as_text).mean()
        score = as_text.map(rates).to_numpy(dtype=float)
    auc = float(roc_auc_score(labels, score))
    return 2 * auc - 1, 2 * hanley_mcneil_se(auc, int(labels.sum()), int((1 - labels).sum()))


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    a = max(auc, 1 - auc)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _matched_reference(
    reference: pd.Series, reference_y: np.ndarray, n: int, kind: str, repeats: int = 5
) -> tuple[float, float] | None:
    if kind == "numeric" or n >= len(reference):
        return signed_association(reference, reference_y, kind)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(repeats):
        rows = rng.choice(len(reference), size=n, replace=False)
        result = signed_association(reference.iloc[rows], reference_y[rows], kind)
        if result is not None:
            draws.append(result)
    if not draws:
        return None
    return float(np.mean([d[0] for d in draws])), float(np.mean([d[1] for d in draws]))


def check_relationships(
    report_id: str,
    current: pd.DataFrame,
    current_y: np.ndarray,
    reference: pd.DataFrame,
    reference_y: np.ndarray,
    contract: DataContract,
    usable: list[str],
    cfg: MonitoringConfig,
) -> list[Finding]:
    findings: list[Finding] = []
    columns = contract.by_name
    for name in usable:
        kind = columns[name].kind
        cur_result = signed_association(current[name], current_y, kind)
        ref_result = _matched_reference(reference[name], reference_y, len(current), kind)
        if ref_result is None or cur_result is None:
            continue
        (ref, ref_se), (cur, cur_se) = ref_result, cur_result
        gain = abs(cur) - abs(ref)
        se = float(np.hypot(ref_se, cur_se)) or 1e-9
        z = gain / se
        leak = gain >= cfg.leakage_strength_gain and z >= cfg.leakage_z
        findings.append(
            make_finding(
                report_id,
                "relationship",
                "label_association_gain",
                feature=name,
                metric_name="association_strength",
                observed=abs(cur),
                reference=abs(ref),
                threshold=abs(ref) + cfg.leakage_strength_gain,
                triggered=leak,
                severity=Severity.HIGH
                if abs(cur) >= cfg.leakage_strength_high
                else Severity.MEDIUM,
                level=EpistemicLevel.STATISTICAL_ANOMALY,
                interpretation=(
                    f"'{name}' alone separates the label with strength {abs(cur):.2f} "
                    f"(reference {abs(ref):.2f}, gain {gain:+.2f}, {z:.1f} standard errors)"
                ),
                evidence={"signed_current": cur, "signed_reference": ref, "gain": gain, "z": z},
            )
        )
        if kind == "numeric":
            flipped = (
                np.sign(cur) != np.sign(ref)
                and min(abs(cur), abs(ref)) >= cfg.reversal_min_strength
            )
            findings.append(
                make_finding(
                    report_id,
                    "relationship",
                    "label_association_reversal",
                    feature=name,
                    metric_name="signed_association",
                    observed=cur,
                    reference=ref,
                    threshold=cfg.reversal_min_strength,
                    triggered=bool(flipped),
                    severity=Severity.MEDIUM,
                    level=EpistemicLevel.STATISTICAL_ANOMALY,
                    interpretation=(
                        f"direction of the '{name}'/label relationship "
                        f"{'reversed' if flipped else 'is unchanged'} ({ref:+.2f} -> {cur:+.2f})"
                    ),
                    evidence={"signed_current": cur, "signed_reference": ref},
                )
            )
    return findings
