# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps for pdfplumber / pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY data/raw/manifest.json ./data/raw/manifest.json

# Runtime env defaults (overridden by docker-compose / ECS task env)
ENV LLM_BACKEND=api \
    DATA_RAW_DIR=data/raw \
    DATA_PROCESSED_DIR=data/processed \
    INDEX_DIR=data/index \
    BM25_TOP_K=50 \
    DENSE_TOP_K=50 \
    RERANK_TOP_K=8 \
    PORT=8000

EXPOSE 8000

# Health check for ALB / docker-compose
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
