"""AEGIS's drift statistics must agree with an independent implementation (Evidently)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from aegis.ingestion.datasets import ADULT
from aegis.monitoring.crosscheck import evidently_drift
from aegis.monitoring.drift import categorical_stats, numeric_stats

COMPLETE = [
    "age",
    "education_num",
    "hours_per_week",
    "capital_gain",
    "capital_loss",
    "marital_status",
    "relationship",
    "race",
    "sex",
]  # features with no missing values


@pytest.fixture(scope="module")
def frames(ctx):
    reference = ctx.reference
    current = ctx.clean(seed=3, n_rows=3000).current.copy()
    current["age"] = current["age"] + 6  # ~0.45 sd shift
    current["hours_per_week"] = current["hours_per_week"] * 1.5
    mask = np.random.default_rng(0).random(len(current)) < 0.3
    current.loc[mask, "marital_status"] = "Never-married"
    spec = ADULT.model_copy(
        update={
            "numeric_features": [f for f in ADULT.numeric_features if f in COMPLETE],
            "categorical_features": [f for f in ADULT.categorical_features if f in COMPLETE],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        external = evidently_drift(spec, reference, current)
    return reference, current, external


@pytest.mark.parametrize("feature", ["age", "education_num", "hours_per_week", "capital_gain"])
def test_numeric_statistics_agree_with_evidently(frames, feature):
    reference, current, external = frames
    ours = numeric_stats(reference[feature].to_numpy(float), current[feature].to_numpy(float), 10)
    theirs = external[feature]
    # Same test, same data: p-values agree up to the exact-vs-asymptotic method.
    assert (ours["p_value"] < 0.01) == (theirs["ks"] < 0.01)
    if theirs["ks"] > 1e-12:
        assert np.log10(ours["p_value"]) == pytest.approx(np.log10(theirs["ks"]), abs=0.5)
    # Wasserstein distance normalised by the reference standard deviation: identical definition.
    assert ours["wasserstein_std"] == pytest.approx(theirs["wasserstein"], rel=0.02, abs=0.005)
    # PSI depends on binning (ours: reference deciles), so compare the verdict band, not the digits.
    assert (ours["psi"] >= 0.1) == (theirs["psi"] >= 0.1) or abs(ours["psi"] - theirs["psi"]) < 0.08


@pytest.mark.parametrize("feature", ["marital_status", "relationship", "sex"])
def test_categorical_statistics_agree_with_evidently(frames, feature):
    reference, current, external = frames
    ours = categorical_stats(reference[feature], current[feature])
    theirs = external[feature]
    assert (ours["p_value"] < 0.01) == (theirs["chisquare"] < 0.01)
    assert ours["psi"] == pytest.approx(theirs["psi"], rel=0.15, abs=0.01)


def test_both_implementations_flag_the_same_features(frames):
    reference, current, external = frames
    flagged_by_evidently = {
        f for f, v in external.items() if v.get("psi", 0) >= 0.1 or v.get("wasserstein", 0) >= 0.1
    }
    ours = set()
    for feature in COMPLETE:
        if pd.api.types.is_numeric_dtype(reference[feature]):
            s = numeric_stats(
                reference[feature].to_numpy(float), current[feature].to_numpy(float), 10
            )
            if s["p_value"] < 0.01 and (
                s["psi"] >= 0.1 or s["wasserstein_std"] >= 0.1 or s["ks_statistic"] >= 0.1
            ):
                ours.add(feature)
        else:
            s = categorical_stats(reference[feature], current[feature])
            if s["p_value"] < 0.01 and (s["psi"] >= 0.1 or s["total_variation"] >= 0.1):
                ours.add(feature)
    assert ours == {"age", "hours_per_week", "marital_status"}
    assert ours <= flagged_by_evidently
