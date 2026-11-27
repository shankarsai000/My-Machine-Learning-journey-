# ==========================================
# Stage 1: Build dependency wheels
# ==========================================

FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN python -m pip wheel \
    --no-cache-dir \
    --wheel-dir=/wheels \
    -r requirements.txt


# ==========================================
# Stage 2: Production runtime
# ==========================================

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies from prebuilt wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --no-index \
    --find-links=/wheels \
    -r requirements.txt \
    && rm -rf /wheels

# Create a non-root user
RUN groupadd --system app \
    && useradd --system --gid app app

# Copy application and trained model
COPY --chown=app:app app ./app
COPY --chown=app:app artifacts ./artifacts

# Run without root privileges
USER app

# Application port
EXPOSE 8080

# Local Docker health check
HEALTHCHECK --interval=30s \
    --timeout=3s \
    --start-period=60s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)" || exit 1

# Start FastAPI
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
