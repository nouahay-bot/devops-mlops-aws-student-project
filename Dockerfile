

# Stage 1: Base Image
# ============================================================================
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="nadaouahay@gmail.com"
LABEL version="1.0"
LABEL description="Iris Classification API - Decision Tree ML Model"
LABEL environment="production"

# Working Directory
WORKDIR /app

# Environment Variables (Build-time)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    FLASK_PORT=5000


# Stage 2: System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel


# Stage 3: Python Dependencies
# ============================================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Stage 4: Application Code
# ============================================================================
COPY api/ /app/api/
COPY model/ /app/model/


# Stage 5: Security & Permissions
# ============================================================================
# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash apiuser && \
    chown -R apiuser:apiuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER apiuser


# Stage 6: Network Configuration
# ============================================================================
# Expose port 5000
EXPOSE 5000


# Stage 7: Health Check
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1


# Stage 8: Entrypoint
# ============================================================================
# Start application with gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "api.app:app"]


