# ─────────────────────────────────────────────────────────────────────
# Phase Retrieval — Production Docker Image
# Multi-stage build: install deps first (cached), then copy source.
# Runs as non-root for security.  Built-in healthcheck.
# ─────────────────────────────────────────────────────────────────────

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir --prefix=/install \
    numpy scipy matplotlib astropy pydantic \
    fastapi "uvicorn[standard]" "sqlalchemy>=2.0" \
    alembic PyJWT python-multipart \
    pydantic-settings httpx psycopg2-binary bcrypt rich scikit-image \
    astroquery

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy source and install the package itself
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY phase_retrieval/ phase_retrieval/
COPY web/ web/
RUN pip install --no-cache-dir --no-deps -e .

# Create runtime directories with proper ownership
RUN mkdir -p /app/data /app/web_outputs && \
    chown -R appuser:appuser /app/data /app/web_outputs /app

# Switch to non-root user
USER appuser

# Healthcheck — hits the liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
