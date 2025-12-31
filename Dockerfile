# Étape 1 : image Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port Flask
EXPOSE 5000

# Commande pour lancer l'API
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "api.main_app:app"]

