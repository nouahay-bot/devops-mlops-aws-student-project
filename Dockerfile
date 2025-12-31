# Stage 1: Base Image
FROM python:3.11-slim AS base

LABEL maintainer="nadaouahay@gmail.com"
LABEL version="1.0"
LABEL description="Iris Classification API - Decision Tree ML Model"
LABEL environment="production"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    FLASK_PORT=5000

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest requests

# Copy app, models, tests
COPY api/ /app/api/
COPY model/ /api/models/
COPY tests/ /app/tests/

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash apiuser && \
    chown -R apiuser:apiuser /app && \
    chmod -R 755 /app
USER apiuser

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main_app:app", "--workers", "2", "--threads", "2", "--worker-class", "gthread", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info"]
