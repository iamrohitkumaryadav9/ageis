#!/bin/sh
# Container start-up: make the instance usable from an empty volume, then serve.
# Every step is idempotent, so restarts are cheap.
set -eu
cd /app

if [ ! -f data/processed/adult/split_manifest.json ]; then
  echo "[aegis] first start: downloading datasets (checksum-verified) and building splits"
  python -m scripts.download_data
  python -m scripts.prepare_data
fi
if [ ! -f artifacts/models/adult/baseline.json ] || [ ! -f artifacts/models/bank/baseline.json ]; then
  echo "[aegis] first start: training baseline models (about a minute on a laptop CPU)"
  python -m scripts.train_baseline
fi
echo "[aegis] applying database migrations"
alembic upgrade head
echo "[aegis] registering reference windows and deployments"
python -m scripts.bootstrap
echo "[aegis] starting API"
exec uvicorn apps.api.main:app_factory --factory --host 0.0.0.0 --port 8000
