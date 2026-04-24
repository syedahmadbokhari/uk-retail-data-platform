# 📊 Retail Data Platform — Production Data Engineering System
### SQL • Python • Airflow • PostgreSQL • dbt • Docker • Scikit-learn • Streamlit • pytest • CI/CD

![CI](https://github.com/syedahmadbokhari/sql-data-analysis/actions/workflows/ci.yml/badge.svg)

A **production-style retail data platform** that covers the full data engineering stack: synthetic event generation, incremental ETL pipeline with watermark tracking, Apache Airflow orchestration, dbt transformation layer, PostgreSQL-ready database, content-based recommendation engine, and a 49-test pytest suite with GitHub Actions CI — all surfaced through an interactive Streamlit dashboard.

Built to simulate a **real-world, always-moving data engineering system** where new sales events arrive continuously, the pipeline processes only what's new, and every run is safe to repeat.

---

## 🚀 Live Dashboard

🔗 **Streamlit App**
https://sql-data-analysis-bisxvwilgc3ntxhken76wy.streamlit.app/

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVENT GENERATION LAYER                       │
│                                                                  │
│  src/data_generator/generate_events.py                          │
│  → Generates N synthetic sales events per run                   │
│  → Appends to fact_sales_events (UUID event_id, forward ts)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐
│  STATIC INGEST      │         │  INCREMENTAL INGEST  │
│  ingest.py          │         │  ingest_events.py    │
│  finance/brands/    │         │  WHERE event_ts >    │
│  info/reviews/      │         │  last watermark      │
│  traffic → raw_*    │         │  → raw_events_agg.   │
└─────────────────────┘         └──────────┬──────────┘
          │                                │
          └────────────────┬───────────────┘
                           ▼
              ┌────────────────────────┐
              │  QUALITY GATE          │
              │  validate_raw_layer()  │
              │  row counts, null rate │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  CLEAN LAYER           │
              │  clean.py              │
              │  raw_* → clean_*       │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  ANALYTICS LAYER       │
              │  aggregate.py          │
              │  clean_* → analytics_* │
              │  + analytics_event_rev │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  dbt (PostgreSQL only) │
              │  staging views +       │
              │  mart tables + tests   │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  QUALITY GATE          │
              │  validate_marts()      │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  FEATURE ENGINEERING   │
              │  features_products     │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  RECOMMENDATION MODEL  │
              │  similarity.pkl        │
              │  cosine similarity     │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  STREAMLIT DASHBOARD   │
              └────────────────────────┘
```

### Data Layer Summary

| Layer | Tables | Purpose |
|-------|--------|---------|
| Events | `fact_sales_events` | Append-only event log — one row per sale |
| Raw | `raw_finance`, `raw_brands`, `raw_info`, `raw_reviews`, `raw_traffic`, `raw_events_aggregated` | Exact source copies + aggregated event data |
| Clean | `clean_finance`, `clean_brands`, `clean_info`, `clean_reviews`, `clean_traffic` | Validated, typed, null-handled |
| Analytics | `analytics_brand_revenue`, `analytics_product_revenue`, `analytics_monthly_traffic`, `analytics_discount_impact`, `analytics_event_revenue` | Pre-computed business metrics |
| Features | `features_products` | ML-ready product feature table |
| Watermarks | `pipeline_watermarks`, `event_ingestion_watermark` | Incremental state tracking |
| Model | `models/similarity.pkl` | Cosine similarity matrix (3,120 × 3,120) |

---

## 📂 Project Structure

```
project/
│
├── .github/workflows/ci.yml          # GitHub Actions — pytest on every push
├── docker-compose.yml                # Airflow + Postgres full stack
├── docker/
│   ├── Dockerfile.airflow            # Airflow image with dbt-postgres
│   └── init-db.sql                   # Creates 'retail' DB on first Postgres boot
│
├── data/
│   └── retailDB.sqlite               # Source DB + all pipeline layers
│
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── staging/                  # Views: stg_finance/brands/info/reviews/traffic
│       └── marts/                    # Tables: mart_brand/product/traffic/discount
│
├── src/
│   ├── data_generator/
│   │   └── generate_events.py        # ★ NEW — synthetic sales event generator
│   ├── utils/
│   │   ├── db.py                     # SQLAlchemy dual-mode + upsert_df()
│   │   ├── logger.py                 # Structured logging
│   │   ├── validation.py             # Row count, null, duplicate checks
│   │   └── watermark.py              # Pipeline watermark tracking
│   ├── etl/
│   │   ├── ingest.py                 # Static source → raw_* (incremental, UPSERT)
│   │   ├── ingest_events.py          # ★ NEW — fact_sales_events → raw_events_aggregated
│   │   ├── clean.py                  # raw_* → clean_*
│   │   └── aggregate.py              # clean_* → analytics_* (incl. event revenue)
│   ├── features/
│   │   └── build_features.py         # features_products table
│   └── recommender.py                # Cosine similarity model
│
├── pipeline/
│   ├── run_pipeline.py               # Local script runner (7 steps)
│   └── dags/
│       └── retail_pipeline.py        # Airflow DAG (10 tasks)
│
├── tests/
│   ├── test_clean.py                 # 24 unit tests
│   ├── test_features.py              # 12 unit tests
│   └── test_recommender.py           # 13 unit tests
│
├── app.py                            # Streamlit dashboard
├── .env.example                      # Credentials template
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the full incremental pipeline
python pipeline/run_pipeline.py

# Or generate events and ingest independently
python -m src.data_generator.generate_events --n 500
python -c "from src.etl.ingest_events import ingest_incremental; ingest_incremental()"

# Launch the dashboard
streamlit run app.py

# Run tests
pytest
```

### Docker (full stack — Airflow + PostgreSQL)

```bash
cp .env.example .env
docker compose up --build
# Airflow UI → http://localhost:8080  (admin / admin)
# Trigger DAG: retail_pipeline
```

---

## 🔄 Synthetic Data Generator

**File:** `src/data_generator/generate_events.py`

Generates realistic retail sales events and appends them to `fact_sales_events`:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | TEXT (UUID) | Unique event identifier |
| `product_id` | TEXT | Drawn from existing product catalogue |
| `price` | REAL | £49.99 – £249.99 (athletic footwear range) |
| `discount` | REAL | 0 – 55% |
| `quantity` | INTEGER | 1 – 5 units |
| `revenue` | REAL | `price × (1 − discount) × quantity` |
| `event_timestamp` | TIMESTAMP | Current time + 0–999 ms forward jitter |

**Key design decision:** timestamps use forward-only jitter (0–999 ms ahead of `NOW()`). This guarantees every batch sits strictly after the previous batch's watermark — the incremental ingest can never miss or double-count events.

```bash
# Generate 200 events (default)
python -m src.data_generator.generate_events

# Generate 500 events with a fixed seed
python -m src.data_generator.generate_events --n 500 --seed 42
```

---

## ⚡ Incremental Pipeline

**File:** `src/etl/ingest_events.py`

Reads only NEW events from `fact_sales_events` since the last successful run:

```
1. Read max_event_ts from event_ingestion_watermark
2. SELECT * FROM fact_sales_events WHERE event_timestamp > max_event_ts
3. Aggregate new events to product level (SUM revenue, AVG price/discount)
4. UPSERT into raw_events_aggregated ON CONFLICT (product_id) DO UPDATE
5. Advance watermark to max(event_timestamp) of processed batch
```

**Idempotency guarantee:** re-running with no new events processes 0 rows and leaves all tables unchanged.

### Demonstrated Results

```
BASELINE   fact_sales_events:    0 rows  |  watermark: none (first run)

RUN 1      generated 200 events  →  processed 200  |  total: 200
           fact_sales_events:  200 rows  |  raw_events_aggregated: 193 products

RUN 2      generated 150 events  →  processed 150  |  total: 350
           fact_sales_events:  350 rows  |  raw_events_aggregated: 333 products

RUN 3      generated 100 events  →  processed 100  |  total: 450
           fact_sales_events:  450 rows  |  raw_events_aggregated: 425 products

RE-RUN     no new events         →  processed   0  (idempotency confirmed ✓)
```

---

## ✈️ Airflow DAG — 10 Tasks

```
[generate_events, ingest_raw] ──► ingest_incremental
                                          │
                                  validate_raw_layer
                                          │
                                    clean_tables
                                          │
                                   build_analytics
                                          │
                                       dbt_run
                                          │
                                   validate_marts
                                          │
                                   build_features
                                          │
                              build_similarity_matrix
```

- `generate_events` and `ingest_raw` run **in parallel** — independent sources
- Two quality gates (`validate_raw_layer`, `validate_marts`) abort the run if checks fail
- `dbt_run` executes `dbt run` + `dbt test` on PostgreSQL, gracefully skips on SQLite/CI
- `retries=2`, `retry_delay=3min` on all tasks

---

## 🗄️ Database Layer

`src/utils/db.py` auto-selects engine based on environment:

```python
# PostgreSQL (Docker / production)
export DB_HOST=postgres DB_NAME=retail DB_USER=airflow DB_PASSWORD=airflow

# SQLite (local dev / CI — zero setup)
# No env vars needed — uses data/retailDB.sqlite automatically
```

`upsert_df()` uses `INSERT ... ON CONFLICT (col) DO UPDATE SET ...` — works on both PostgreSQL and SQLite 3.24+ with automatic unique index creation.

---

## 🗂️ dbt Transformation Layer

Staging views clean raw data in SQL (PostgreSQL only):

| Model | Source | Key transformation |
|-------|--------|-------------------|
| `stg_finance` | `raw_finance` | Cast to float, clip discount [0,1] |
| `stg_brands` | `raw_brands` | `INITCAP(TRIM(brand))` |
| `stg_reviews` | `raw_reviews` | `REPLACE(',','.')` for European decimals |
| `stg_traffic` | `raw_traffic` | Strip whitespace |

Mart tables aggregate to business metrics — with `not_null`, `unique`, and `accepted_values` schema tests.

```bash
dbt run   --profiles-dir ./dbt --project-dir ./dbt
dbt test  --profiles-dir ./dbt --project-dir ./dbt
```

---

## 🧮 Recommendation System

Content-based filtering using cosine similarity on a 6-feature normalised product vector.

**Features:** `brand_encoded`, `listing_price`, `discount`, `revenue`, `rating`, `review_count`

**Example** — query: *Women's adidas Running Ultraboost 19 Shoes*

| Product | Similarity |
|---------|-----------|
| Men's adidas Running Ultraboost 19 Shoes | 99.8% |
| Women's adidas Running Ultraboost 19 Shoes | 99.8% |
| Men's adidas Running Ultraboost 20 Shoes | 99.6% |

---

## 🧪 Testing — 49 tests, all passing

```bash
pytest
# 49 passed
```

| File | Tests | What's covered |
|------|-------|----------------|
| `test_clean.py` | 24 | European decimal bug, discount clipping, null drops |
| `test_features.py` | 12 | Column structure, brand encoding, null imputation |
| `test_recommender.py` | 13 | Self-exclusion, sort order, score range, edge cases |

---

## ⚙️ Technology Stack

| Layer | Tools |
|-------|-------|
| **Event Generation** | Python, UUID, Pandas |
| **Database** | PostgreSQL (production), SQLite (local / CI) |
| **ORM / Connections** | SQLAlchemy 2.x, psycopg2 |
| **ETL** | Python, Pandas |
| **Transformations** | dbt-core, dbt-postgres |
| **Orchestration** | Apache Airflow 2.8 (DAG + PythonOperator) |
| **Containerisation** | Docker Compose |
| **ML / Recommendations** | scikit-learn (StandardScaler, cosine_similarity) |
| **Testing** | pytest, unittest.mock |
| **CI/CD** | GitHub Actions |
| **Dashboard** | Streamlit, Power BI |
| **Logging** | Python logging (console + file) |

---

## 💼 Skills Demonstrated

**Data Engineering**
- Incremental pipeline with watermark state tracking
- Synthetic event generation simulating real data streams
- Idempotent UPSERT pattern safe for production re-runs
- ETL with staging / clean / analytics / feature layers
- Apache Airflow DAG with parallel tasks, retries, quality gates
- PostgreSQL + SQLite via SQLAlchemy — environment-driven selection
- dbt staging views + mart tables with schema tests
- Docker Compose: Airflow + PostgreSQL full-stack deployment

**Software Engineering**
- Modular Python package structure with clean separation of concerns
- 49-test pytest suite with mocking, fixtures, edge-case coverage
- GitHub Actions CI — automated testing on every push

**Machine Learning & Analytics**
- Feature engineering, label encoding, StandardScaler normalisation
- Content-based recommendation with cosine similarity
- Advanced SQL — CTEs, window functions, aggregations

---

## 👨‍💻 Author

**Ahmad Bokhari**
