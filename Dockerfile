FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY api/ api/
COPY tests/ tests/

# Exposer le port
EXPOSE 5000

# Commande par défaut
CMD ["python", "api/main_app.py"]
