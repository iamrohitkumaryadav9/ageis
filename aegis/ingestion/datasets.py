"""Reproducible dataset acquisition and normalisation.

Raw archives are downloaded once, verified against a pinned SHA-256 and kept
untouched under ``data/raw/<name>/``. Everything downstream is derived from the
verified archive, so a corrupted or silently changed upstream file fails loudly
instead of shifting every experiment.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel

TARGET = "target"


class DatasetSpec(BaseModel):
    name: str
    url: str
    sha256: str
    archive_name: str
    numeric_features: list[str]
    categorical_features: list[str]
    sensitive_attributes: list[str]
    dropped_columns: dict[str, str]  # column -> documented reason
    positive_label: str
    citation: str

    @property
    def features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


ADULT = DatasetSpec(
    name="adult",
    url="https://archive.ics.uci.edu/static/public/2/adult.zip",
    sha256="7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
    archive_name="adult.zip",
    numeric_features=["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"],
    categorical_features=[
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ],
    sensitive_attributes=["age", "race", "sex", "native_country", "marital_status"],
    dropped_columns={
        "fnlwgt": "census sampling weight, not a property of the individual",
        "education": "exact duplicate of education_num in categorical form",
    },
    positive_label=">50K",
    citation="Becker, B. & Kohavi, R. (1996). Adult. UCI ML Repository. doi:10.24432/C5XW20",
)

BANK = DatasetSpec(
    name="bank",
    url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    sha256="e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
    archive_name="bank-marketing.zip",
    numeric_features=["age", "balance", "day", "campaign", "pdays", "previous"],
    categorical_features=[
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ],
    sensitive_attributes=["age", "marital", "job"],
    dropped_columns={
        "duration": "call duration is only known after the outcome; using it is target leakage "
        "(documented by the dataset authors)",
    },
    positive_label="yes",
    citation="Moro, S., Rita, P. & Cortez, P. (2014). Bank Marketing. UCI ML Repository. "
    "doi:10.24432/C5K306",
)

DATASETS: dict[str, DatasetSpec] = {ADULT.name: ADULT, BANK.name: BANK}

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


class ChecksumMismatchError(RuntimeError):
    """The downloaded archive does not match the pinned checksum."""


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}") from None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: DatasetSpec, raw_dir: Path, *, force: bool = False) -> Path:
    """Fetch the raw archive (idempotent) and write a provenance record next to it."""
    target_dir = raw_dir / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / spec.archive_name
    if force or not archive.exists():
        with httpx.stream("GET", spec.url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            tmp = archive.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
            tmp.replace(archive)
    actual = sha256_file(archive)
    if actual != spec.sha256:
        raise ChecksumMismatchError(
            f"{spec.name}: sha256 {actual} does not match pinned {spec.sha256}; "
            "the upstream file changed or the download is corrupt"
        )
    provenance = target_dir / "provenance.json"
    if force or not provenance.exists():
        provenance.write_text(
            json.dumps(
                {
                    "dataset": spec.name,
                    "source_url": spec.url,
                    "sha256": actual,
                    "bytes": archive.stat().st_size,
                    "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "citation": spec.citation,
                    "license": "CC BY 4.0 (UCI Machine Learning Repository)",
                },
                indent=2,
            )
            + "\n"
        )
    return archive


def _load_adult(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as zf:
        frames = []
        for member, skip in (("adult.data", 0), ("adult.test", 1)):  # .test has a banner line
            frames.append(
                pd.read_csv(
                    io.BytesIO(zf.read(member)),
                    names=_ADULT_COLUMNS,
                    skiprows=skip,
                    skipinitialspace=True,
                    na_values="?",
                )
            )
    df = pd.concat(frames, ignore_index=True).dropna(subset=["income"])
    income = df["income"].str.rstrip(".")  # the test split labels end with a period
    df[TARGET] = (income == ADULT.positive_label).astype("int8")
    return df.drop(columns=["income"])


def _load_bank(archive: Path) -> pd.DataFrame:
    with (
        zipfile.ZipFile(archive) as outer,
        zipfile.ZipFile(io.BytesIO(outer.read("bank.zip"))) as inner,
    ):
        df = pd.read_csv(io.BytesIO(inner.read("bank-full.csv")), sep=";")
    df[TARGET] = (df["y"] == BANK.positive_label).astype("int8")
    return df.drop(columns=["y"])


_LOADERS = {"adult": _load_adult, "bank": _load_bank}


def load_raw(spec: DatasetSpec, raw_dir: Path) -> pd.DataFrame:
    """Return the normalised modelling table: declared features + binary ``target``."""
    archive = raw_dir / spec.name / spec.archive_name
    if not archive.exists():
        raise FileNotFoundError(f"{archive} missing; run `python -m scripts.download_data` first")
    df = _LOADERS[spec.name](archive)
    df = df.drop(columns=list(spec.dropped_columns))
    df = df[[*spec.features, TARGET]].reset_index(drop=True)
    for col in spec.numeric_features:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    for col in spec.categorical_features:
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
    return df
