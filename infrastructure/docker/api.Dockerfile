# AEGIS API image: locked Python environment + application code. CPU only.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lockfile changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY aegis ./aegis
COPY apps/__init__.py ./apps/__init__.py
COPY apps/api ./apps/api
COPY scripts ./scripts
COPY alembic.ini README.md ./
COPY data/runbooks ./data/runbooks
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 aegis && mkdir -p /app/data /app/artifacts /app/mlruns \
    && chown -R aegis:aegis /app/data /app/artifacts /app/mlruns
USER aegis
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=240s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
ENTRYPOINT ["sh", "/app/scripts/docker_entrypoint.sh"]
