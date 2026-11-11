"""Service-level errors, mapped 1:1 to HTTP responses by the API layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ROLES = ("viewer", "operator", "approver")


@dataclass(frozen=True)
class Principal:
    identity: str
    role: str

    def at_least(self, role: str) -> bool:
        return self.role in ROLES and ROLES.index(self.role) >= ROLES.index(role)


SYSTEM = Principal("system", "operator")


class ServiceError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message, self.details = message, details


class NotFoundError(ServiceError):
    status, code = 404, "not_found"


class ConflictError(ServiceError):
    status, code = 409, "conflict"


class ForbiddenError(ServiceError):
    status, code = 403, "forbidden"


class InvalidInputError(ServiceError):
    status, code = 422, "invalid_input"


class UnavailableError(ServiceError):
    status, code = 503, "unavailable"
