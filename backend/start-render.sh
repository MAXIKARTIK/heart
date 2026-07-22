#!/bin/sh
# Startup script for single-instance online deployment (Render / Koyeb / etc.).
# Runs DB migrations, starts a Celery worker in-process only if a broker is
# configured, then runs the API in the foreground. Referenced by render.yaml's
# dockerCommand. Not used by local docker-compose or the default image command.
set -e

# Apply database migrations (PostgreSQL in production; harmless on SQLite).
alembic upgrade head

# Start the Celery worker in the background only when a broker is configured.
# "solo" pool keeps memory low enough for small free instances.
if [ -n "$REDIS_URL" ] || [ -n "$CELERY_BROKER_URL" ]; then
  echo "Starting Celery worker (broker configured)..."
  celery -A app.celery_app:celery_app worker \
    --pool=solo --without-gossip --without-mingle --without-heartbeat \
    --loglevel=warning &
else
  echo "No broker configured; batch requests will run synchronously."
fi

# Run the API in the foreground (keeps the container alive; honors $PORT).
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
