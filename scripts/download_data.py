"""Download and checksum-verify the public datasets. Usage: uv run python -m scripts.download_data"""

from __future__ import annotations

import argparse

from aegis.config import get_settings
from aegis.ingestion.datasets import DATASETS, download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    settings = get_settings()
    for spec in DATASETS.values():
        path = download(spec, settings.raw_dir, force=args.force)
        print(f"{spec.name}: verified sha256 {spec.sha256[:16]}...  {path}")


if __name__ == "__main__":
    main()
