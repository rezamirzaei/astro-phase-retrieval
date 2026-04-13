# Phase Retrieval — Web Application

Full-stack web interface for astronomical wavefront sensing and X-ray
crystallography built with **FastAPI** (REST + WebSocket API),
**Angular 18** (SPA frontend), **SQLAlchemy 2.0** (ORM with connection
pooling), **Alembic** (migrations), and **Docker Compose** (deployment).

## Quick Start

### Docker Compose (recommended)

```bash
docker compose up --build
```

Open **http://localhost** → register → explore the demo.

| Service  | URL                                |
|----------|------------------------------------|
| Frontend | http://localhost                    |
| API      | http://localhost:8000               |
| Docs     | http://localhost:8000/docs          |
| DB       | postgresql://phase:phase@localhost:5432/phase |

### Local Development

```bash
# Install Python deps
pip install -e ".[web,dev]"

# Start backend
uvicorn web.main:app --reload --port 8000

# In another terminal: start Angular dev server
cd web/frontend && npm install && npm start
```

Frontend runs on http://localhost:4200 and proxies `/api` to :8000.

## Architecture

```
┌──────────────┐     ┌───────────────────────┐     ┌──────────┐
│  Angular 18  │◄───►│  FastAPI + JWT         │◄───►│ SQLite / │
│  (nginx)     │     │  (uvicorn)             │     │ Postgres │
│  port 80     │     │  port 8000             │     │          │
└──────────────┘     └───────────┬────────────┘     └──────────┘
                          │             │
                   ┌──────┴──────┐  ┌───┴────────────┐
                   │  src.*      │  │  Job Queue      │
                   │  algorithms │  │  ThreadPool     │
                   │  optics     │  │  WebSocket      │
                   │  metrics    │  │  Progress       │
                   │  viz        │  │  Streaming      │
                   └─────────────┘  └────────────────┘
```

## API Endpoints

### Auth
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| POST   | /api/auth/register                    | Create account                |
| POST   | /api/auth/login                       | Get access + refresh tokens   |
| POST   | /api/auth/refresh                     | Refresh token pair            |
| GET    | /api/auth/me                          | Current user info             |

### Data Management
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/data/presets                     | List download presets          |
| POST   | /api/data/download/{key}              | Download MAST preset           |
| GET    | /api/data/fits                        | List data files (paginated)    |
| POST   | /api/data/upload                      | Upload custom FITS/NPY file    |
| POST   | /api/data/synthetic                   | Generate synthetic PSF         |

### Algorithms
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/algorithms/                      | List algorithms with defaults  |
| POST   | /api/algorithms/run                   | Run single algorithm           |
| POST   | /api/algorithms/compare               | Compare all algorithms         |
| GET    | /api/algorithms/benchmark/cases       | List benchmark cases           |
| POST   | /api/algorithms/benchmark             | Run synthetic benchmark suite  |

### Results
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/results/                         | List results (paginated)       |
| GET    | /api/results/dashboard                | Dashboard stats                |
| GET    | /api/results/{id}                     | Single result details          |
| GET    | /api/results/{id}/plots/{name}        | Serve plot PNG                 |
| GET    | /api/results/{id}/artifacts/{name}    | Parsed artifact content        |
| GET    | /api/results/{id}/export              | Download result as ZIP         |
| POST   | /api/results/export-batch             | Batch export multiple jobs     |
| DELETE | /api/results/{id}                     | Delete result                  |

### Background Jobs (v3.0)
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/v1/jobs/                         | List all tracked jobs          |
| GET    | /api/v1/jobs/{job_id}                 | Poll job status + progress     |
| WS     | /api/ws/jobs/{job_id}?token=JWT       | Real-time progress streaming   |

### Crystallography
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/crystallography/presets          | List COD crystal presets       |
| POST   | /api/crystallography/download/{key}   | Download CIF from COD          |
| GET    | /api/crystallography/cif-files        | List available CIF files       |
| POST   | /api/crystallography/simulate         | Simulate diffraction pattern   |
| POST   | /api/crystallography/run              | Run crystallographic retrieval |
| POST   | /api/crystallography/compare          | Compare algorithms on crystal  |
| GET    | /api/crystallography/{id}             | Get crystal result             |
| GET    | /api/crystallography/{id}/plots/{n}   | Serve crystal plot PNG         |
| DELETE | /api/crystallography/{id}             | Delete crystal result          |

### Studies
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| POST   | /api/studies/validation-campaign      | Run multi-obs validation       |
| GET    | /api/studies/validation-campaigns/{id}/artifacts/{name} | Campaign artifact |

### Educational
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/explain/algorithms               | Algorithm explanations         |
| GET    | /api/explain/metrics                  | Metric explanations            |
| GET    | /api/explain/science                  | Science overview               |

### Health & Observability
| Method | Path                                  | Description                   |
|--------|---------------------------------------|-------------------------------|
| GET    | /api/health                           | Liveness probe (uptime, ver)   |
| GET    | /api/readiness                        | Readiness: DB + disk checks    |
| GET    | /api/version                          | API version + Python info      |

## Features

- **10 phase-retrieval algorithms** — GS, ER, HIO, RAAR, WF, DR, ADMM, FISTA, Sparse PR, PINN
- **User authentication** — JWT register/login/refresh, bcrypt hashing, audit logging
- **Data management** — download real HST observations, generate synthetic PSFs, upload custom files
- **Run & compare algorithms** — select data + algorithm + parameters, see plots and metrics
- **Background job queue** — non-blocking compute with WebSocket real-time progress streaming
- **Crystallography** — parse CIF files from COD, simulate diffraction, run phase retrieval
- **Results gallery** — browse (paginated), view detail plots, export as ZIP, batch export
- **Educational** — learn about algorithms, metrics, and the science
- **Dashboard** — stats, recent results, quick-action buttons
- **Security** — rate-limited login, refresh tokens, security headers, request-ID tracing
- **Observability** — structured logging with request ID, liveness/readiness probes
- **Connection pooling** — `pool_pre_ping`, configurable pool size for PostgreSQL
- **Graceful shutdown** — drains job queue, disposes DB engine
- **Interactive API docs** — Swagger UI at `/docs`, ReDoc at `/redoc`, enriched OpenAPI tags

## Configuration (Environment Variables)

| Variable                     | Default                          | Description                   |
|------------------------------|----------------------------------|-------------------------------|
| `PR_SECRET_KEY`              | `dev-only-change-me-...`         | JWT signing secret            |
| `PR_ADMIN_PASSWORD`          | `admin123`                       | Seed admin password           |
| `PR_DATABASE_URL`            | `sqlite:///./web/phase_...db`    | SQLAlchemy DB URL             |
| `PR_CORS_ORIGINS`            | localhost:4532,4533              | Comma-separated CORS origins  |
| `PR_MAX_CONCURRENT_JOBS`     | `4`                              | Thread-pool size for jobs     |
| `PR_SHUTDOWN_TIMEOUT_SECONDS`| `30`                             | Graceful shutdown wait        |
| `PR_UPLOAD_MAX_BYTES`        | `104857600` (100 MB)             | File upload size limit        |

## Technology Stack

| Layer       | Technology                                |
|-------------|-------------------------------------------|
| Frontend    | Angular 18, Material Design, TypeScript   |
| Backend     | FastAPI, Pydantic v2, SQLAlchemy 2.0      |
| Auth        | JWT (PyJWT), bcrypt, refresh tokens       |
| Real-time   | WebSocket (Starlette), asyncio.Queue      |
| Job Queue   | ThreadPoolExecutor, ContextVars           |
| Database    | SQLite (dev) / PostgreSQL 16 (Docker)     |
| Migration   | Alembic                                   |
| Deploy      | Docker Compose, nginx reverse proxy       |
| Lint        | ruff, mypy (strict)                       |
| Test        | pytest + httpx TestClient                 |

