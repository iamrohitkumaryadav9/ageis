"""Register default reference windows and initial deployments (idempotent).

Usage: uv run python -m scripts.bootstrap      (after `alembic upgrade head`)
"""

from __future__ import annotations

import json

from aegis.config import get_settings
from aegis.services.core import AegisService
from aegis.services.errors import SYSTEM
from aegis.storage.db import Database


def main() -> None:
    settings = get_settings().model_copy(update={"llm_enabled": False})
    service = AegisService(settings, Database(settings.database_url))
    try:
        print(json.dumps(service.bootstrap(SYSTEM), indent=2, default=str))
    finally:
        service.shutdown()
        service.db.dispose()


if __name__ == "__main__":
    main()
