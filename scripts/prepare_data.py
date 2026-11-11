"""Build the reproducible train/validation/test/production splits.

Usage: uv run python -m scripts.prepare_data [--seed N]
"""

from __future__ import annotations

import argparse

from aegis.config import get_settings
from aegis.ingestion.datasets import DATASETS, load_raw
from aegis.ingestion.splits import make_splits, read_manifest, write_splits


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.seed)
    args = parser.parse_args()
    for spec in DATASETS.values():
        df = load_raw(spec, settings.raw_dir)
        splits = make_splits(df, args.seed)
        write_splits(spec.name, args.seed, splits, settings.processed_dir)
        manifest = read_manifest(spec.name, settings.processed_dir)
        print(
            f"{spec.name}: rows={len(df)} sizes={manifest.sizes} pos_rate={manifest.positive_rate}"
        )


if __name__ == "__main__":
    main()
