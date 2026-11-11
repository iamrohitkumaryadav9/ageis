import numpy as np
import pandas as pd
import pytest

from aegis.ingestion.contracts import derive_contract
from aegis.ingestion.datasets import (
    ADULT,
    BANK,
    TARGET,
    ChecksumMismatchError,
    download,
    get_spec,
    load_raw,
)
from aegis.ingestion.splits import FRACTIONS, build_manifest, frame_fingerprint, make_splits


def test_adult_loads_expected_shape_and_labels(settings):
    download(ADULT, settings.raw_dir)
    df = load_raw(ADULT, settings.raw_dir)
    assert len(df) == 48842  # 32561 train + 16281 test rows in the original release
    assert list(df.columns) == [*ADULT.features, TARGET]
    assert set(df[TARGET].unique()) == {0, 1}
    # The '.'-suffixed labels of adult.test must be folded into the same classes.
    assert df[TARGET].mean() == pytest.approx(0.2393, abs=1e-3)
    assert "fnlwgt" not in df and "education" not in df
    # '?' placeholders become real missing values, not a category called '?'.
    assert df["workclass"].isna().sum() > 0 and "?" not in set(df["workclass"].dropna())


def test_bank_drops_documented_leakage_column(settings):
    download(BANK, settings.raw_dir)
    df = load_raw(BANK, settings.raw_dir)
    assert len(df) == 45211
    assert "duration" not in df.columns
    assert "duration" in BANK.dropped_columns


def test_download_rejects_checksum_mismatch(settings, tmp_path):
    tampered = ADULT.model_copy(update={"sha256": "0" * 64})
    (tmp_path / "adult").mkdir()
    (tmp_path / "adult" / ADULT.archive_name).write_bytes(b"not the dataset")
    with pytest.raises(ChecksumMismatchError):
        download(tampered, tmp_path)


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        get_spec("imagenet")


def test_splits_are_disjoint_exhaustive_and_stratified(adult_splits):
    indices = [set(part.index) for part in adult_splits.values()]
    total = sum(len(i) for i in indices)
    assert len(set().union(*indices)) == total == 48842  # no row in two splits, none lost
    for name, part in adult_splits.items():
        assert len(part) / total == pytest.approx(FRACTIONS[name], abs=0.001)
        assert part[TARGET].mean() == pytest.approx(0.2393, abs=0.002)


def test_splits_are_reproducible_and_seed_sensitive(settings):
    df = load_raw(ADULT, settings.raw_dir)
    a, b, c = make_splits(df, 7), make_splits(df, 7), make_splits(df, 8)
    assert build_manifest("adult", 7, a) == build_manifest("adult", 7, b)
    assert frame_fingerprint(a["train"]) == frame_fingerprint(b["train"])
    assert build_manifest("adult", 8, c).index_sha256 != build_manifest("adult", 7, a).index_sha256


def test_contract_captures_reference_schema(adult_splits):
    contract = derive_contract(ADULT, adult_splits["validation"])
    cols = contract.by_name
    assert set(cols) == set(ADULT.features)
    assert cols["age"].kind == "numeric" and cols["age"].minimum >= 17
    assert cols["sex"].categories == ["Female", "Male"]
    assert 0.04 < cols["workclass"].nullable_rate < 0.08
    # Adult has natural duplicates once fnlwgt is dropped; the contract must record that
    # baseline so duplicate monitoring compares against it instead of against zero.
    assert contract.reference_duplicate_rate > 0.01


def test_contract_version_changes_with_schema(adult_splits):
    ref = adult_splits["validation"]
    base = derive_contract(ADULT, ref)
    changed = ref.copy()
    changed["sex"] = np.where(changed["sex"] == "Male", "M", "F")
    assert derive_contract(ADULT, changed).version != base.version
    assert derive_contract(ADULT, ref.copy()).version == base.version


def test_frame_fingerprint_is_order_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert frame_fingerprint(df) != frame_fingerprint(df.iloc[::-1])
