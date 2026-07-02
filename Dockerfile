# =============================================================================
# Multi-stage Dockerfile for Redrob AI Ranker
# Stage 1: Build Next.js frontend static export
# Stage 2: Production Python API server + static frontend
# =============================================================================

# --- Stage 1: Build frontend ---
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --production=false
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production ---
FROM python:3.11-slim AS production

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SentenceTransformer model to cache in Docker image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY rank.py ./

# Copy built frontend
COPY --from=frontend-build /app/frontend/out ./frontend/out

# Copy data and artifacts (if present)
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Create jobs directory
RUN mkdir -p ./config/jobs

# Environment defaults
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV ARTIFACTS_DIR=./artifacts
ENV DATA_DIR=./data
ENV CONFIG_DIR=./config
ENV LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
