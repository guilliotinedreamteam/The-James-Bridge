# syntax=docker/dockerfile:1

# ============================================================
# NeuroBridge Docker Image
# Multi-stage build for smaller production images
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Production ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY neurobridge/ ./neurobridge/
COPY neurobridge.py .

# Create model directory
RUN mkdir -p /app/models

# Environment defaults
ENV NB_API_HOST=0.0.0.0
ENV NB_API_PORT=5000
ENV NB_MODEL_DIR=/app/models
ENV LOG_LEVEL=INFO

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

# Default command: start API server
CMD ["python", "neurobridge.py", "serve"]
