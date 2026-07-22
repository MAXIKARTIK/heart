# Backend image: FastAPI API + Celery worker share this image.
# Build context is the PROJECT ROOT (see docker-compose.yml) so we can install
# the shared heart_ml package and train the model inside the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Shared ML library (its deps: scikit-learn / pandas / numpy / joblib).
COPY pyproject.toml ./
COPY heart_ml ./heart_ml
RUN pip install --upgrade pip && pip install -e .

# 2) API dependencies.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# 3) Data + application code.
COPY heart_2020_cleaned.csv ./
COPY backend ./backend

# 4) Train the model at build time so the serialized artifact is guaranteed to
#    match the scikit-learn version installed in this image.
RUN python -m heart_ml.train --sample 40000 --no-plots

WORKDIR /app/backend
EXPOSE 8000

# Default: run the API. The worker service overrides this in docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
