# ====================================================
# Memory API - API Service Dockerfile
# ====================================================
# Multi-stage build for optimized image size
# Uses Python 3.11-slim for production

# ====================================================
# STAGE 1: Builder
# ====================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ====================================================
# STAGE 2: Runtime
# ====================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# ====================================================
# Environment Configuration
# ====================================================
# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Unbuffered Python output for Docker logs
ENV PYTHONUNBUFFERED=1

# Disable pip version check in container
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ====================================================
# Application Code
# ====================================================
# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# ====================================================
# Optional: Pre-download Models
# ====================================================
# Set PRELOAD_MODELS=true to bake models into the image
# This makes the image larger but startup faster
ARG PRELOAD_MODELS=false
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    fi

# ====================================================
# Expose Port & Health Check
# ====================================================
EXPOSE 8888

# Health check - waits for startup (models loading can take time)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# ====================================================
# Default Command
# ====================================================
# Run the API server
# Can be overridden in docker-compose or docker run
CMD ["python", "-m", "src.main"]
