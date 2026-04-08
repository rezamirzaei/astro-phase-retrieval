# Phase Retrieval — Web Application

Full-stack web interface for astronomical wavefront sensing built with
**FastAPI** (REST API), **Angular 18** (SPA frontend), **SQLAlchemy 2.0** (ORM),
**Alembic** (migrations), and **Docker Compose** (deployment).

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
┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  Angular 18  │◄───►│  FastAPI + JWT    │◄───►│ SQLite / │
│  (nginx)     │     │  (uvicorn)        │     │ Postgres │
│  port 80     │     │  port 8000        │     │          │
└──────────────┘     └──────────────────┘     └──────────┘
                            │
                     ┌──────┴──────┐
                     │  src.*      │
                     │  algorithms │
                     │  optics     │
                     │  metrics    │
                     │  viz        │
                     └─────────────┘
```

## API Endpoints

| Method | Path                         | Description                   |
|--------|------------------------------|-------------------------------|
| POST   | /api/auth/register           | Create account                |
| POST   | /api/auth/login              | Get JWT token                 |
| GET    | /api/auth/me                 | Current user info             |
| GET    | /api/data/presets            | List download presets          |
| POST   | /api/data/download/{key}     | Download MAST preset           |
| GET    | /api/data/fits               | List available data files      |
| POST   | /api/data/synthetic          | Generate synthetic PSF         |
| GET    | /api/algorithms/             | List algorithms                |
| POST   | /api/algorithms/run          | Run single algorithm           |
| POST   | /api/algorithms/compare      | Compare all algorithms         |
| GET    | /api/results/                | List user's results            |
| GET    | /api/results/dashboard       | Dashboard stats                |
| GET    | /api/results/{id}            | Single result details          |
| GET    | /api/results/{id}/plots/{n}  | Serve plot PNG                 |
| DELETE | /api/results/{id}            | Delete result                  |
| GET    | /api/explain/algorithms      | Algorithm explanations         |
| GET    | /api/explain/metrics         | Metric explanations            |
| GET    | /api/explain/science         | Science overview               |
| GET    | /api/health                  | Liveness probe                 |

## Features

- **User authentication** — JWT-based register/login/logout
- **Data management** — download real HST observations, generate synthetic PSFs
- **Run algorithms** — select data + algorithm + parameters, see plots and metrics
- **Compare algorithms** — run all 7+ algorithms on the same data, side-by-side
- **Results gallery** — browse, view detail plots, delete old runs
- **Educational** — learn about algorithms, metrics, and the science
- **Dashboard** — stats, recent results, quick-action buttons
- **Interactive API docs** — Swagger UI at `/docs`, ReDoc at `/redoc`

## Technology Stack

| Layer     | Technology                                |
|-----------|-------------------------------------------|
| Frontend  | Angular 18, Material Design, TypeScript   |
| Backend   | FastAPI, Pydantic v2, SQLAlchemy 2.0      |
| Auth      | JWT (python-jose), PBKDF2-SHA256          |
| Database  | SQLite (dev) / PostgreSQL 16 (Docker)     |
| Migration | Alembic                                   |
| Deploy    | Docker Compose, nginx reverse proxy       |
| Lint      | ruff, mypy (strict)                       |
| Test      | pytest + httpx TestClient                 |

