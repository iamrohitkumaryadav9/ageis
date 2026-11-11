"""Request dependencies: the service singleton and bearer-token authentication."""

from __future__ import annotations

import hmac
from collections.abc import Callable

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aegis.config import Settings
from aegis.services.core import AegisService
from aegis.services.errors import ROLES, ForbiddenError, Principal, ServiceError

_bearer = HTTPBearer(auto_error=False)


class UnauthenticatedError(ServiceError):
    status, code = 401, "unauthenticated"


def parse_tokens(raw: str) -> dict[str, Principal]:
    """``token:identity:role`` triples, comma separated. Malformed entries are a startup error."""
    tokens: dict[str, Principal] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        pieces = entry.split(":")
        if len(pieces) != 3 or pieces[2] not in ROLES or len(pieces[0]) < 16:
            raise ValueError(
                "AEGIS_API_TOKENS entries must look like '<token of 16+ chars>:<identity>:<viewer|operator|approver>'"
            )
        tokens[pieces[0]] = Principal(identity=pieces[1], role=pieces[2])
    return tokens


def get_service(request: Request) -> AegisService:
    return request.app.state.service


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_principal(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)
) -> Principal:
    tokens: dict[str, Principal] = request.app.state.tokens
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthenticatedError("a bearer token is required")
    presented = credentials.credentials.encode()
    match = None
    for token, principal in tokens.items():  # constant-time comparison against every token
        if hmac.compare_digest(token.encode(), presented):
            match = principal
    if match is None:
        raise UnauthenticatedError("invalid token")
    request.state.principal = match
    return match


def require(role: str) -> Callable[[Principal], Principal]:
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.at_least(role):
            raise ForbiddenError(f"the {role} role is required", role=principal.role)
        return principal

    return dependency
