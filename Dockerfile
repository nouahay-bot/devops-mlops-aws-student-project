# ---------------------------
# Dockerfile pour ml-api
# ---------------------------

# Utiliser une image Python légère
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'API
COPY api/ ./api/
COPY api/models/model.pkl /app/model/
COPY api/models/scaler.pkl /app/model/
COPY tests/ ./tests/

# Exposer le port Flask
EXPOSE 5000

# Définir la variable d'environnement pour Flask
ENV FLASK_APP=api/main_app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Commande pour lancer Flask
CMD ["flask", "run"]
