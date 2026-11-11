"""Feature-level diagnostics: *how* did a distribution change?

The central question for a drifted numeric feature is whether the change is
additive (a location shift) or multiplicative (a unit/scale error). Both models
are fitted to the quantile-quantile relationship between reference and current
data; whichever leaves little of the change unexplained names the pattern.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

Pattern = Literal["no_change", "scale", "shift", "ambiguous", "shape_change", "category_shift"]


class FeatureInspection(BaseModel):
    feature: str
    kind: Literal["numeric", "categorical"]
    pattern: Pattern
    n_reference: int
    n_current: int
    # numeric
    scale_factor: float | None = None
    shift_offset: float | None = None
    scale_unexplained: float | None = None  # share of the quantile change the scale model misses
    shift_unexplained: float | None = None
    change_in_sd: float | None = None
    zero_fraction_reference: float | None = None
    zero_fraction_current: float | None = None
    quantiles: dict[str, list[float]] = Field(default_factory=dict)
    max_correlation_change: float | None = None
    correlation_partner: str | None = None
    # categorical
    category_shifts: dict[str, float] = Field(default_factory=dict)
    new_categories: list[str] = Field(default_factory=list)
    summary: str = ""


def _fit(reference: np.ndarray, current: np.ndarray) -> dict:
    probs = np.linspace(0.02, 0.98, 49)
    r, c = np.quantile(reference, probs), np.quantile(current, probs)
    sd = float(np.std(reference)) or 1.0
    total = float(np.mean(np.abs(c - r)))
    out = {
        "change_in_sd": total / sd,
        "scale_factor": None,
        "shift_offset": None,
        "scale_unexplained": None,
        "shift_unexplained": None,
    }
    if total / sd < 0.08:  # within sampling noise of the quantiles; below any drift threshold
        return {**out, "pattern": "no_change"}
    offset = float(np.median(c - r))
    shift_un = float(np.mean(np.abs(c - r - offset))) / total
    scale_un, factor = float("inf"), None
    nonzero = r != 0
    if nonzero.sum() >= 5:
        factor = float(np.median(c[nonzero] / r[nonzero]))
        scale_un = float(np.mean(np.abs(c - factor * r))) / total
    else:
        # Zero-inflated feature: most quantiles are 0 for both samples, so fit the factor on
        # the non-zero values. A rescale keeps zeros at zero; a shift would have moved them.
        ref_nz, cur_nz = reference[reference != 0], current[current != 0]
        zeros_kept = abs((reference == 0).mean() - (current == 0).mean()) < 0.05
        if zeros_kept and len(ref_nz) >= 30 and len(cur_nz) >= 30:
            rq, cq = np.quantile(ref_nz, probs), np.quantile(cur_nz, probs)
            ratios = cq[rq != 0] / rq[rq != 0]
            if ratios.size >= 5:
                factor = float(np.median(ratios))
                spread = float(np.mean(np.abs(cq - rq))) or 1.0
                scale_un = float(np.mean(np.abs(cq - factor * rq))) / spread
    out.update(
        shift_offset=offset,
        shift_unexplained=shift_un,
        scale_factor=factor,
        scale_unexplained=None if factor is None else scale_un,
    )
    scale_ok = (
        factor is not None and factor > 0 and scale_un < 0.25 and abs(np.log(factor)) >= np.log(1.3)
    )
    shift_ok = shift_un < 0.25
    if scale_ok and (not shift_ok or scale_un < shift_un / 2):
        pattern = "scale"
    elif shift_ok and (not scale_ok or shift_un < scale_un / 2):
        pattern = "shift"
    elif scale_ok and shift_ok:
        pattern = "ambiguous"
    else:
        pattern = "shape_change"
    return {**out, "pattern": pattern}


def inspect_feature(
    feature: str,
    kind: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
) -> FeatureInspection:
    ref, cur = reference[feature].dropna(), current[feature].dropna()
    if kind == "categorical":
        ref_p = ref.astype(str).value_counts(normalize=True)
        cur_p = cur.astype(str).value_counts(normalize=True)
        delta = cur_p.sub(ref_p, fill_value=0.0)
        movers = delta.abs().sort_values(ascending=False).index[:5]
        new = [c for c in cur_p.index if c not in ref_p.index][:5]
        moved = float(delta.abs().sum() / 2)
        return FeatureInspection(
            feature=feature,
            kind="categorical",
            pattern="category_shift" if moved >= 0.05 else "no_change",
            n_reference=len(ref),
            n_current=len(cur),
            category_shifts={_clip(c): round(float(delta[c]), 4) for c in movers},
            new_categories=[_clip(c) for c in new],
            summary=f"{moved:.1%} of probability mass moved between categories",
        )

    r = ref.to_numpy(dtype=float)
    c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < 20:
        return FeatureInspection(
            feature=feature,
            kind="numeric",
            pattern="shape_change",
            n_reference=len(r),
            n_current=len(c),
            summary="too few finite current values to characterise",
        )
    fit = _fit(r, c)
    probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    corr_change, partner = _correlation_change(feature, reference, current, numeric_features)
    pattern = fit["pattern"]
    if pattern == "scale":
        summary = (
            f"current ~= reference x {fit['scale_factor']:.4g} across quantiles (multiplicative)"
        )
    elif pattern == "shift":
        summary = f"current ~= reference {fit['shift_offset']:+.4g} across quantiles (additive)"
    else:
        summary = {
            "no_change": "quantiles match the reference",
            "ambiguous": "an additive and a multiplicative change fit equally well",
            "shape_change": "neither a pure shift nor a pure rescaling explains the change",
        }[pattern]
    return FeatureInspection(
        feature=feature,
        kind="numeric",
        pattern=fit["pattern"],
        n_reference=len(r),
        n_current=len(c),
        scale_factor=fit["scale_factor"],
        shift_offset=fit["shift_offset"],
        scale_unexplained=fit["scale_unexplained"],
        shift_unexplained=fit["shift_unexplained"],
        change_in_sd=fit["change_in_sd"],
        zero_fraction_reference=float((r == 0).mean()),
        zero_fraction_current=float((c == 0).mean()),
        quantiles={
            "p": probs,
            "reference": np.quantile(r, probs).round(4).tolist(),
            "current": np.quantile(c, probs).round(4).tolist(),
        },
        max_correlation_change=corr_change,
        correlation_partner=partner,
        summary=summary,
    )


def _clip(value: object, limit: int = 40) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(value))[:limit]


def _correlation_change(
    feature: str, reference: pd.DataFrame, current: pd.DataFrame, numeric_features: list[str]
) -> tuple[float | None, str | None]:
    """Largest change in Spearman correlation between ``feature`` and any other numeric feature.

    Rank correlation is unchanged by shifting or rescaling a column, so a large value
    here means row-level relationships were disturbed, not just the marginal."""
    others = [
        f
        for f in numeric_features
        if f != feature and f in current.columns and pd.api.types.is_numeric_dtype(current[f])
    ]
    if not others or not pd.api.types.is_numeric_dtype(current[feature]):
        return None, None
    ref_corr = reference[[feature, *others]].corr(method="spearman")[feature].drop(feature)
    cur_corr = current[[feature, *others]].corr(method="spearman")[feature].drop(feature)
    delta = (cur_corr - ref_corr).abs().dropna()
    if delta.empty:
        return None, None
    return float(delta.max()), str(delta.idxmax())
