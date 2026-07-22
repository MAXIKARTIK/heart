"""Lightweight optional API-key authentication.

Auth is only enforced when ``API_KEY`` is configured. This keeps local
development frictionless while allowing the deployment to lock the API down by
simply setting the environment variable. See main.py for a startup warning when
running in production without a key.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return  # auth disabled
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key (send it in the 'X-API-Key' header).",
        )
