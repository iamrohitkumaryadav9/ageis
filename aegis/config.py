"""Central runtime configuration.

Every tunable lives here and is overridable through ``AEGIS_*`` environment
variables (or a ``.env`` file), so the same code runs in tests, the benchmark,
local development and Docker Compose without edits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    # --- paths -----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"

    # --- persistence -----------------------------------------------------
    database_url: str = "postgresql+psycopg://aegis:aegis@localhost:5432/aegis"

    # --- security --------------------------------------------------------
    # Comma separated ``token:identity:role`` triples. Roles: viewer, operator, approver.
    # There is deliberately no default approver token: approval is impossible until one is set.
    api_tokens: str = ""
    max_upload_bytes: int = 20 * 1024 * 1024
    max_upload_rows: int = 200_000
    # Separation of duties: the approver must differ from whoever created the remediation.
    require_distinct_approver: bool = False

    # --- local LLM (Ollama) ---------------------------------------------
    llm_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_fallback_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.1
    ollama_max_tokens: int = 700
    ollama_timeout_s: float = 120.0
    ollama_keep_alive: str = "2m"
    ollama_max_retries: int = 1
    ollama_num_thread: int | None = None

    # --- runbook retrieval ----------------------------------------------
    # lexical | hybrid. Hybrid adds local embeddings; see docs/decisions.md (D10) for the measurement.
    rag_method: str = "lexical"

    # --- agent bounds ----------------------------------------------------
    max_tool_calls: int = Field(default=5, ge=0, le=5)
    tool_timeout_s: float = 30.0

    # --- MLOps / observability ------------------------------------------
    mlflow_tracking_uri: str = f"sqlite:///{REPO_ROOT / 'artifacts' / 'mlflow.db'}"
    mlflow_enabled: bool = True
    otlp_endpoint: str = ""  # e.g. http://localhost:4318 ; empty => in-process span store only
    log_level: str = "INFO"
    log_json: bool = True

    # --- reproducibility -------------------------------------------------
    seed: int = 20260101

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runbook_dir(self) -> Path:
        return self.data_dir / "runbooks"

    @property
    def scenario_dir(self) -> Path:
        return self.data_dir / "scenarios"

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
