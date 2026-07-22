"""Celery application.

The app is always constructed so that ``tasks.py`` can decorate tasks at import
time. When no real broker is configured we fall back to an in-memory broker with
``task_always_eager`` so calling ``.delay()`` executes inline -- this keeps the
code path identical whether or not Redis is available.
"""
from __future__ import annotations

from celery import Celery

from .core.config import settings

_broker = settings.effective_broker or "memory://"
_backend = settings.effective_backend or "cache+memory://"

celery_app = Celery("heart", broker=_broker, backend=_backend)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_always_eager=not settings.celery_enabled,
    task_store_eager_result=True,
    result_expires=3600,
)

# Ensure task modules are registered with this app.
celery_app.autodiscover_tasks(["app"])
