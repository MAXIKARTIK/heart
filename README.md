# Heart Disease Risk Prediction & Clinical Risk Analysis

An end-to-end machine learning system that predicts coronary heart-disease risk
from clinical indicators. It takes the exploratory Kaggle notebook (Decision
Tree + K-Nearest Neighbors on the CDC *Personal Key Indicators* dataset),
modernizes it for the current Python/scikit-learn stack, and wraps the model in
a production-style service: a **FastAPI** REST API with request validation,
prediction logging to **PostgreSQL**, asynchronous batch scoring via
**Celery + Redis**, and a **React + TypeScript** frontend, all orchestrated with
**Docker Compose**.

> Educational/portfolio project. Not a medical device and not a substitute for
> professional diagnosis.

---

## Architecture

```
                    ┌──────────────────────────┐
                    │  React + TS + Vite (SPA)  │
                    │  TanStack Query           │
                    └────────────┬─────────────┘
                                 │  /api/v1/*  (nginx proxy)
                                 ▼
   ┌───────────────────────────────────────────────────────┐
   │                    FastAPI  (api)                       │
   │  Pydantic validation · /predict · /batch · /health      │
   │  ┌───────────────┐   loads   ┌──────────────────────┐   │
   │  │ PredictionSvc │ ◀──────── │ model.joblib bundle  │   │
   │  └───────────────┘           │ (sklearn Pipeline +  │   │
   │        │                     │  tuned threshold)    │   │
   │        │ log                 └──────────────────────┘   │
   └────────┼───────────────────────────┬───────────────────┘
            ▼                            │ enqueue batch
   ┌─────────────────┐          ┌────────▼────────┐   ┌─────────────┐
   │  PostgreSQL     │          │  Redis (broker) │◀─▶│ Celery worker│
   │  predictions    │          └─────────────────┘   │ batch_predict│
   │  (SQLAlchemy /  │◀───────────────────────────────┤  logs results│
   │   Alembic)      │                                 └─────────────┘
   └─────────────────┘

   Shared library `heart_ml` (installed into api + worker) guarantees identical
   feature engineering / preprocessing at train time and serve time.
```

## Tech stack

| Layer            | Technologies |
|------------------|--------------|
| Machine Learning | scikit-learn, pandas, NumPy, joblib, Matplotlib, Seaborn |
| Backend          | FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic, Celery, Redis |
| Database         | PostgreSQL (SQLite for zero-infra local dev) |
| Frontend         | React, TypeScript, Vite, TanStack Query |
| DevOps           | Docker, Docker Compose, nginx |

## Repository layout

```
heart/
├── heart_ml/                     # Shared ML library (train + serve)
│   ├── config.py                 # Feature lists + exact category values
│   ├── data.py                   # Load dataset / extract target
│   ├── features.py               # FeatureEngineer transformer
│   ├── pipeline.py               # ColumnTransformer + DT/KNN pipelines
│   ├── train.py                  # Train, tune, compare, serialize
│   └── artifacts/                # model.joblib, metrics.json, plots/
├── backend/                      # FastAPI service
│   ├── app/
│   │   ├── main.py               # App factory + lifespan
│   │   ├── core/                 # config (pydantic-settings), security
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── db.py / models.py     # SQLAlchemy engine + Prediction table
│   │   ├── ml.py                 # Model-serving layer
│   │   ├── celery_app.py/tasks.py# Async batch scoring
│   │   └── routers/              # health, predict, batch
│   ├── alembic/                  # DB migrations
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                     # React + TS + Vite SPA
├── heart-disease-prediction-other.ipynb   # Modernized EDA + modeling notebook
├── heart_2020_cleaned.csv        # CDC dataset (~320k rows)
├── pyproject.toml                # Installs the heart_ml package
└── docker-compose.yml
```

---

## The ML pipeline

The dataset is highly imbalanced (~8.5% positive). Accuracy alone is misleading
(predicting "no disease" for everyone scores ~91%), so the pipeline is built
around that reality:

- **Preprocessing** (leak-free, inside a single `Pipeline`): feature engineering
  → `StandardScaler` on numeric features → `OneHotEncoder(handle_unknown="ignore")`
  on categoricals. Preprocessing is fit on training folds only.
- **Feature engineering**: `TotalUnhealthyDays` (physical + mental unwell days),
  `SleepDeviation` (distance from ~7h), `IsObese` (BMI ≥ 30).
- **Class imbalance**: `class_weight="balanced"` for the Decision Tree; the
  decision **threshold is tuned to maximize F1** on a validation split instead of
  defaulting to 0.5.
- **Tuning**: `GridSearchCV` (scored on PR-AUC / average precision).
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC on a held-out 20% test
  set drawn from the full dataset.

### Results (tuned thresholds, held-out test set)

| Model         | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------|----------|-----------|--------|-------|---------|
| **Decision Tree** (deployed) | 0.851 | 0.288 | 0.501 | **0.366** | **0.814** |
| KNN           | 0.848 | 0.273 | 0.468 | 0.345 | 0.795 |

The Decision Tree is selected and serialized (it wins on F1/ROC-AUC and is tiny
and fast to serve). Comparison plots are in `heart_ml/artifacts/plots/`:

![Model comparison](heart_ml/artifacts/plots/model_comparison.png)

> For reference, the un-tuned baselines in the original notebook (0.5 threshold,
> no class weighting) get F1 ≈ 0.20 (KNN) and ≈ 0.24 (DT) — the balanced,
> threshold-tuned pipeline roughly doubles recall and F1.

---

## Quickstart A — local (no Docker)

Requires Python 3.10+ and Node 18+. Uses SQLite and runs batch jobs
synchronously (no Redis needed).

```bash
# 1) Install the ML library + train the model
pip install -e .
python -m heart_ml.train            # writes heart_ml/artifacts/model.joblib

# 2) Run the API
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs

# 3) Run the frontend (in another terminal)
cd frontend && npm install && npm run dev
# app at http://localhost:5173  (Vite proxies /api -> :8000)
```

## Quickstart B — full stack with Docker Compose

Brings up PostgreSQL, Redis, the API, a Celery worker, and the frontend. The
model is trained during the image build.

```bash
cp .env.example .env      # then edit POSTGRES_PASSWORD, etc.
docker compose up --build
# frontend  -> http://localhost:3000
# API docs  -> http://localhost:8000/docs
```

---

## API reference

Base path: `/api/v1`

| Method | Path                | Description |
|--------|---------------------|-------------|
| GET    | `/health`           | Liveness + whether the model/async are available |
| GET    | `/health/model`     | Model metadata and metrics |
| POST   | `/predict`          | Score one patient (logged to DB) |
| POST   | `/batch`            | Score many (async via Celery, or inline if no broker) |
| GET    | `/batch/{task_id}`  | Poll an async batch job |

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"BMI":40,"Smoking":"Yes","AlcoholDrinking":"No","Stroke":"Yes",
       "PhysicalHealth":20,"MentalHealth":10,"DiffWalking":"Yes","Sex":"Male",
       "AgeCategory":"80 or older","Race":"White","Diabetic":"Yes",
       "PhysicalActivity":"No","GenHealth":"Poor","SleepTime":4,
       "Asthma":"Yes","KidneyDisease":"Yes","SkinCancer":"Yes"}'
# -> {"prediction":1,"risk_label":"At risk","probability":0.97,"threshold":0.71,...}
```

**Security note:** authentication is off by default for local dev. Set `API_KEY`
to require an `X-API-Key` header on `/predict` and `/batch`. The API logs a
warning if started with `ENVIRONMENT=production` and no key.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///./heart.db` | SQLAlchemy connection string |
| `REDIS_URL` | *(empty)* | Enables Celery; empty → synchronous batch |
| `API_KEY` | *(empty)* | If set, required via `X-API-Key` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `HEART_TRAIN_SAMPLE` | `40000` | Stratified training-set cap |

## Notebook

`heart-disease-prediction-other.ipynb` is the original Kaggle EDA + modeling
notebook, updated to run on Python 3.12 with current scikit-learn / pandas /
seaborn, and with the test-set data-leakage bug fixed (the scaler/encoder are no
longer re-fit on the test data). Re-run it top to bottom to reproduce the EDA
and baseline models.

---

## Notes on scope & resume alignment

This deployment focuses on the two algorithms in this notebook — **Decision Tree
and KNN**. A couple of honest callouts:

- The resume bullet mentions comparing five models (Logistic Regression, Decision
  Tree, Random Forest, SVM, KNN). This repo ships **DT + KNN**. Two ways to make
  them consistent: (a) reword the bullet to "compared tree-based and
  instance-based models (Decision Tree, KNN)", or (b) extend the comparison —
  `heart_ml/pipeline.py` and `train.py` are model-agnostic, so adding LR/RF/SVM
  is a small change (a few lines each). Ask and it can be added.
- The LLM/RAG parts of the resume stack (LangChain, ChromaDB, NVIDIA NIM) are
  intentionally **not** shoehorned in here — they belong to the agentic projects,
  not a tabular clinical classifier.
```
