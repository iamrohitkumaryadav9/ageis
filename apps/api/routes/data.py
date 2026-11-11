"""Data ingestion and monitoring."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from aegis.services.core import AegisService
from aegis.services.errors import InvalidInputError, Principal
from apps.api.deps import get_service, require
from apps.api.schemas import (
    BatchOut,
    IncidentOut,
    MonitoringRunOut,
    MonitorRequest,
    MonitorResponse,
)

router = APIRouter(tags=["data"])


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes; refuse larger uploads without buffering them."""
    chunks, size = [], 0
    while chunk := await upload.read(1 << 20):
        size += len(chunk)
        if size > limit:
            raise InvalidInputError(f"upload exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def _ingest(
    role: str, dataset: str, file: UploadFile, service: AegisService, principal: Principal
) -> BatchOut:
    if not (file.filename or "").lower().endswith(".csv"):
        raise InvalidInputError("only .csv uploads are accepted")
    payload = await _read_limited(file, service.settings.max_upload_bytes)
    frame = service.parse_csv(payload)
    # The client-supplied filename is recorded as text only; it never becomes a path.
    source = "upload:" + "".join(c for c in (file.filename or "")[:60] if c.isalnum() or c in "._-")
    return BatchOut.model_validate(
        service.register_batch(principal, dataset, role, frame, source=source)
    )


@router.post("/data/reference", response_model=BatchOut, status_code=201)
async def upload_reference(
    dataset: str = Form(max_length=40),
    file: UploadFile = File(...),
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> BatchOut:
    return await _ingest("reference", dataset, file, service, principal)


@router.post("/data/current", response_model=BatchOut, status_code=201)
async def upload_current(
    dataset: str = Form(max_length=40),
    file: UploadFile = File(...),
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> BatchOut:
    return await _ingest("current", dataset, file, service, principal)


@router.post("/monitor/run", response_model=MonitorResponse)
def monitor(
    body: MonitorRequest,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> MonitorResponse:
    out = service.run_monitoring(principal, body.dataset, body.batch_id)
    return MonitorResponse(
        run=MonitoringRunOut.model_validate(out["run"]),
        incident=IncidentOut.model_validate(out["incident"]) if out["incident"] else None,
        replayed=out["replayed"],
    )
