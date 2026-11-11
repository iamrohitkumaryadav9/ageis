"""Covariate drift: per-feature two-sample comparison of current vs reference.

Methods and what each assumes
-----------------------------
* Kolmogorov-Smirnov (numeric): non-parametric test on the largest ECDF gap.
  Assumes i.i.d. samples and continuous data; with ties (integer or
  zero-inflated features) its p-value is conservative. Most sensitive around
  the median, weak in the tails.
* Chi-square homogeneity (categorical): requires adequate expected counts, so
  rare categories are pooled. Sensitive to sample size, blind to magnitude.
* Population Stability Index: an effect size (symmetrised KL over bins), no
  sampling distribution. Depends on binning; conventional bands 0.1 / 0.25.
* Wasserstein-1 distance (numeric): "how far mass must move", reported in
  reference standard deviations so features are comparable. Sensitive to
  outliers and to pure scale changes, which KS/PSI saturate on.

P-values are Benjamini-Hochberg adjusted across features. A feature is flagged
only if the adjusted p-value is significant AND an effect size is material:
significance alone is not operational importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from aegis.ingestion.contracts import DataContract
from aegis.monitoring.common import benjamini_hochberg, make_finding, psi
from aegis.monitoring.config import MonitoringConfig
from aegis.schemas.findings import EpistemicLevel, Finding, Severity, SkippedCheck


def numeric_psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    ref_counts = np.bincount(
        np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1
    )
    cur_counts = np.bincount(
        np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1
    )
    return psi(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())


def numeric_stats(reference: np.ndarray, current: np.ndarray, bins: int) -> dict:
    ks = ks_2samp(reference, current, method="asymp")
    scale = float(np.std(reference)) or 1.0
    return {
        "ks_statistic": float(ks.statistic),
        "p_value": float(ks.pvalue),
        "psi": numeric_psi(reference, current, bins),
        "wasserstein_std": float(wasserstein_distance(reference, current)) / scale,
        "reference_mean": float(np.mean(reference)),
        "current_mean": float(np.mean(current)),
        "reference_std": float(np.std(reference)),
        "current_std": float(np.std(current)),
        "reference_median": float(np.median(reference)),
        "current_median": float(np.median(current)),
    }


def categorical_stats(reference: pd.Series, current: pd.Series) -> dict:
    ref_counts = reference.astype(str).value_counts()
    cur_counts = current.astype(str).value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref = ref_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    cur = cur_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    # Pool categories whose expected count in the smaller sample is < 5 (chi-square validity).
    pooled_share = (ref + cur) / (ref.sum() + cur.sum())
    rare = pooled_share * min(ref.sum(), cur.sum()) < 5
    if rare.sum() > 1:
        ref = np.append(ref[~rare], ref[rare].sum())
        cur = np.append(cur[~rare], cur[rare].sum())
    table = np.vstack([ref, cur])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        chi2, p_value, v = 0.0, 1.0, 0.0
    else:
        chi2, p_value, _, _ = chi2_contingency(table, correction=False)
        v = float(np.sqrt(chi2 / table.sum()))  # 2 x k table => min(r-1, c-1) = 1
    ref_p, cur_p = ref / ref.sum(), cur / cur.sum()
    shift = (
        cur_counts.reindex(categories, fill_value=0) / cur_counts.sum()
        - ref_counts.reindex(categories, fill_value=0) / ref_counts.sum()
    )
    top = shift.abs().sort_values(ascending=False).index[:3]
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": v,
        "psi": psi(ref_p, cur_p),
        "total_variation": float(0.5 * np.abs(ref_p - cur_p).sum()),
        "largest_shifts": {str(c)[:40]: round(float(shift[c]), 4) for c in top},
    }


def check_drift(
    report_id: str,
    current: pd.DataFrame,
    reference: pd.DataFrame,
    contract: DataContract,
    usable: list[str],
    cfg: MonitoringConfig,
) -> tuple[list[Finding], list[SkippedCheck]]:
    skipped: list[SkippedCheck] = []
    stats: dict[str, dict] = {}
    columns = contract.by_name
    for name in usable:
        ref, cur = reference[name].dropna(), current[name].dropna()
        if len(cur) < cfg.min_samples or len(ref) < cfg.min_samples:
            skipped.append(
                SkippedCheck(
                    monitor="drift",
                    check="distribution",
                    feature=name,
                    reason=f"only {len(cur)} non-null current rows (< {cfg.min_samples})",
                )
            )
            continue
        if columns[name].kind == "numeric":
            cur_values = cur.to_numpy(dtype=float)
            cur_values = cur_values[np.isfinite(cur_values)]
            stats[name] = {
                "kind": "numeric",
                **numeric_stats(ref.to_numpy(dtype=float), cur_values, cfg.psi_bins),
            }
        else:
            stats[name] = {"kind": "categorical", **categorical_stats(ref, cur)}

    names = list(stats)
    adjusted = benjamini_hochberg([stats[n]["p_value"] for n in names])
    findings: list[Finding] = []
    for name, p_adj in zip(names, adjusted, strict=True):
        s = stats[name]
        significant = p_adj < cfg.drift_alpha
        if s["kind"] == "numeric":
            effects = {
                "psi": s["psi"] >= cfg.psi_medium,
                "ks": s["ks_statistic"] >= cfg.ks_effect,
                "wasserstein": s["wasserstein_std"] >= cfg.wasserstein_effect,
            }
            test = "ks"
        else:
            effects = {
                "psi": s["psi"] >= cfg.psi_medium,
                "cramers_v": s["cramers_v"] >= cfg.cramers_v_effect,
                "total_variation": s["total_variation"] >= cfg.tvd_effect,
            }
            test = "chi2"
        material = any(effects.values())
        triggered = significant and material
        findings.append(
            make_finding(
                report_id,
                "drift",
                f"{s['kind']}_drift",
                feature=name,
                metric_name="psi",
                observed=s["psi"],
                reference=0.0,
                threshold=cfg.psi_medium,
                triggered=triggered,
                severity=Severity.HIGH if s["psi"] >= cfg.psi_high else Severity.MEDIUM,
                level=EpistemicLevel.STATISTICAL_ANOMALY,
                interpretation=_interpret(name, s, p_adj, significant, material, test),
                evidence={
                    **{k: v for k, v in s.items() if k != "kind"},
                    "p_adjusted": p_adj,
                    "test": test,
                    "significant": significant,
                    # per-method verdicts, so methods can be compared on identical data
                    "method_flags": {
                        test: s["p_value"] < cfg.drift_alpha,
                        f"{test}_bh": significant,
                        **{f"{k}_effect": v for k, v in effects.items()},
                    },
                    "n_reference": len(reference[name].dropna()),
                    "n_current": len(current[name].dropna()),
                },
            )
        )
    return findings, skipped


def _interpret(name, s, p_adj, significant, material, test) -> str:
    head = f"'{name}': PSI {s['psi']:.3f}, {test} BH-adjusted p={p_adj:.2g}"
    if s["kind"] == "numeric":
        head += (
            f", KS D={s['ks_statistic']:.3f}, Wasserstein {s['wasserstein_std']:.2f} sd, "
            f"mean {s['reference_mean']:.4g} -> {s['current_mean']:.4g}"
        )
    else:
        head += f", Cramer's V {s['cramers_v']:.3f}"
    if significant and material:
        return head + ". Distribution differs from reference by a material amount."
    if significant:
        return (
            head + ". Statistically detectable but below every effect-size threshold; not flagged."
        )
    return head + ". No significant difference."
