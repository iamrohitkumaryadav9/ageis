"""Train, evaluate, register and track the baseline models.

Usage: uv run python -m scripts.train_baseline [--dataset adult|bank] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json

from aegis.config import get_settings
from aegis.ingestion.contracts import derive_contract
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.models.training import train_baseline
from aegis.observability.tracking import log_run


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    if args.no_mlflow:
        settings = settings.model_copy(update={"mlflow_enabled": False})
    registry = ModelRegistry(settings.model_dir)

    for name in args.dataset or sorted(DATASETS):
        spec = DATASETS[name]
        splits = {
            s: read_split(name, s, settings.processed_dir) for s in ("train", "validation", "test")
        }
        contract = derive_contract(spec, splits["validation"])
        contract_path = settings.model_dir / name / "contract.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(contract.model_dump_json(indent=2) + "\n")

        result = train_baseline(spec, splits, contract, registry, settings.seed)
        print(f"\n=== {name}: selected {result.selected.version} ===")
        print(f"reason: {result.selection_reason}")
        for algo, record in result.records.items():
            test = record.metrics["test"]
            print(
                f"  {algo:<24} cv_auc={record.metrics['cv']['roc_auc_mean']:.4f} "
                f"test: acc={test['accuracy']:.4f} prec={test['precision']:.4f} "
                f"rec={test['recall']:.4f} f1={test['f1']:.4f} auc={test['roc_auc']:.4f} "
                f"brier={test['brier']:.4f} ece={test['ece']:.4f} thr={record.threshold:.2f} "
                f"fit={record.metrics['training_seconds']:.2f}s "
                f"p95_row={record.metrics['latency']['single_row_p95_ms']:.2f}ms"
            )
            run_id = log_run(
                settings,
                experiment=f"aegis-baseline-{name}",
                run_name=record.version,
                params={
                    **record.params,
                    "algorithm": algo,
                    "seed": record.seed,
                    "threshold": record.threshold,
                    "contract_version": record.contract_version,
                },
                metrics=record.metrics,
                tags={
                    "selected": str(record.version == result.selected.version),
                    "artifact_sha256": record.artifact_sha256,
                },
                artifacts=[registry.artifact_path(name, record.version).parent / "record.json"],
                sklearn_model=registry.load(name, record.version),
                registered_model_name=f"aegis-{name}",
            )
            print(f"    mlflow run: {run_id}")
        print(
            "  confusion (test):", json.dumps(result.selected.metrics["test"]["confusion_matrix"])
        )


if __name__ == "__main__":
    main()
