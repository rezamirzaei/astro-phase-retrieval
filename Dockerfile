# --- Backend: Python + FastAPI ---
FROM python:3.11-slim AS backend

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir numpy scipy matplotlib astropy pydantic && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" "sqlalchemy>=2.0" \
        alembic PyJWT python-multipart \
        pydantic-settings httpx psycopg2-binary bcrypt rich scikit-image \
        astroquery

# Copy source and install
COPY src/ src/
COPY phase_retrieval/ phase_retrieval/
COPY web/ web/
RUN pip install --no-cache-dir -e .

# Create runtime directories
RUN mkdir -p /app/data /app/web_outputs

EXPOSE 8000
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
