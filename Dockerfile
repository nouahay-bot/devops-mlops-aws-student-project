# Stage 1: Base Image
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="nadaouahay@gmail.com"
LABEL version="1.0"
LABEL description="Iris Classification API - Decision Tree ML Model"
LABEL environment="production"

# Working Directory
WORKDIR /app

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    FLASK_PORT=5000

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest

# Application code
COPY api/ /app/api/
COPY model/ /app/model/
COPY tests/ /app/tests/

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash apiuser \
    && chown -R apiuser:apiuser /app \
    && chmod -R 755 /app
USER apiuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Start application with gunicorn (point to main_app.py)
CMD ["gunicorn",
     "--bind", "0.0.0.0:5000",
     "--workers", "2",
     "--threads", "2",
     "--worker-class", "gthread",
     "--timeout", "60",
     "--access-logfile", "-",
     "--error-logfile", "-",
     "--log-level", "info",
     "api.main_app:app"]
