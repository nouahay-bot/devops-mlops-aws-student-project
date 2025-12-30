# 🌸 Iris Classification API \- DevOps-MLOps Pipeline

**Pipeline DevOps-MLOps complet** pour l'entraînement, la conteneurisation et le déploiement d'un modèle de machine learning en production sur AWS.

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)  
- [Architecture](#architecture)  
- [Prérequis](#prérequis)  
- [Installation](#installation)  
- [Utilisation](#utilisation)  
- [API Endpoints](#api-endpoints)  
- [Tests](#tests)  
- [Déploiement](#déploiement)  
- [Structure du projet](#structure-du-projet)  
- [Contribution](#contribution)

---

## 🎯 Vue d'ensemble

Ce projet démontre un pipeline **MLOps-DevOps moderne** complet :

✅ **Phase 1 \- MLOps** : Entraînement d'un modèle Decision Tree sur le dataset Iris  
✅ **Phase 2 \- DevOps** : Conteneurisation avec Docker et automatisation CI/CD  
✅ **Phase 3 \- Infrastructure** : Déploiement sur AWS EC2

### Cas d'usage

Prédiction de l'espèce d'une fleur Iris basée sur 4 caractéristiques (longueur/largeur des sépales et pétales).

### Modèle utilisé

- **Type** : Decision Tree Classifier  
- **Accuracy** : 96.67%  
- **Precision** : 96.97%  
- **Recall** : 96.67%  
- **Dataset** : Iris (150 observations, 3 classes, 4 features)

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────┐

│                 PIPELINE DEVOPS-MLOPS COMPLET               │

└─────────────────────────────────────────────────────────────┘

PHASE 1: MLOps (Entraînement)

─────────────────────────────

AWS SageMaker Notebook

    ↓

Dataset Iris

    ↓

Entraînement Decision Tree

    ↓

Sauvegarde model.pkl \+ scaler.pkl

PHASE 2: DevOps (Développement & CI/CD)

────────────────────────────────────────

Code source (app.py, requirements.txt)

    ↓

GitHub Repository

    ↓

GitHub Actions CI/CD

    ↓

Build Docker Image (425 MB)

    ↓

Push DockerHub/ECR

PHASE 3: Déploiement (Infrastructure)

──────────────────────────────────────

AWS EC2 Instance (t2.micro)

    ↓

Docker Run

    ↓

Flask API (Port 5000\)

    ↓

Prédictions en Production ✓

---

## 📦 Prérequis

### Outils requis

- **Docker** \>= 20.10  
- **Python** \>= 3.11  
- **Git**  
- **AWS Account** (pour déploiement EC2)

### Installation des outils

**Windows (PowerShell):**

\# Installer Docker Desktop

choco install docker-desktop \-y

\# Vérifier l'installation

docker \--version

docker ps

**macOS:**

brew install docker

docker \--version

**Linux (Ubuntu):**

sudo apt-get update

sudo apt-get install docker.io \-y

sudo usermod \-aG docker $USER

---

## 🚀 Installation

### 1\. Cloner le repository

git clone https://github.com/username/devops-mlops-aws-student-project.git

cd devops-mlops-aws-student-project

### 2\. Installer les dépendances Python (local)

\# Créer un environnement virtuel

python \-m venv venv

\# Activer l'environnement

\# Windows

venv\\Scripts\\activate

\# macOS/Linux

source venv/bin/activate

\# Installer les dépendances

pip install \-r requirements.txt

### 3\. Vérifier les fichiers du modèle

ls model/

\# Devrait contenir :

\# \- model.pkl (1.2 MB)

\# \- scaler.pkl (0.8 KB)

---

## 💻 Utilisation

### Option 1 : Exécution locale (Python)

\# Activer l'environnement virtuel

source venv/bin/activate  \# macOS/Linux

\# ou

venv\\Scripts\\activate  \# Windows

\# Lancer l'API

python \-m api

\# L'API démarre sur http://localhost:5000

### Option 2 : Exécution avec Docker (Recommandé)

\# Build l'image Docker

docker build \-t ml-api:1.0 .

\# Lancer le conteneur

docker run \-d \\

  \-p 5000:5000 \\

  \--name ml-api-container \\

  \--restart unless-stopped \\

  ml-api:1.0

\# Vérifier que le conteneur tourne

docker ps

\# Voir les logs

docker logs \-f ml-api-container

### Option 3 : Docker Compose (si disponible)

\# Lancer les services

docker-compose up \-d

\# Arrêter les services

docker-compose down

---

## 🔌 API Endpoints

### 1\. Health Check

GET /health

**Response (200 OK):**

{

  "status": "healthy",

  "model\_loaded": true,

  "timestamp": "2024-12-24T10:30:45.123456"

}

**Utilité** : Vérifier que l'API répond (health checks, load balancers)

---

### 2\. Prédiction (Endpoint principal)

POST /predict

Content-Type: application/json

{

  "features": \[5.1, 3.5, 1.4, 0.2\]

}

**Response (200 OK):**

{

  "prediction": "Setosa",

  "class\_id": 0,

  "probabilities": {

    "Setosa": 0.99,

    "Versicolor": 0.01,

    "Virginica": 0.0

  },

  "confidence": 0.99,

  "timestamp": "2024-12-24T10:31:20.654321"

}

**Parameters:**

- `features` (array\[4\]) : 4 nombres float  
  - `features[0]` : Sepal length (cm)  
  - `features[1]` : Sepal width (cm)  
  - `features[2]` : Petal length (cm)  
  - `features[3]` : Petal width (cm)

**Returns:**

- `prediction` : Classe prédite ("Setosa", "Versicolor", "Virginica")  
- `class_id` : ID de la classe (0, 1, ou 2\)  
- `probabilities` : Probabilités pour chaque classe  
- `confidence` : Confiance de la prédiction (max des probabilités)

---

### 3\. Informations API

GET /info

**Response (200 OK):**

{

  "app\_name": "Iris Classification API",

  "version": "1.0.0",

  "model\_type": "Decision Tree Classifier",

  "dataset": "Iris Dataset",

  "classes": \["Setosa", "Versicolor", "Virginica"\],

  "num\_features": 4,

  "feature\_names": \[

    "sepal length (cm)",

    "sepal width (cm)",

    "petal length (cm)",

    "petal width (cm)"

  \],

  "endpoints": {

    "GET /health": "Vérifier l'état du service",

    "POST /predict": "Faire une prédiction",

    "GET /info": "Informations sur l'API"

  }

}

---

## 🧪 Tests

### Test 1 : Health Check

curl http://localhost:5000/health

**Expected:** Status 200, model\_loaded: true

---

### Test 2 : Prédiction correcte (Setosa)

curl \-X POST http://localhost:5000/predict \\

  \-H "Content-Type: application/json" \\

  \-d '{"features": \[5.1, 3.5, 1.4, 0.2\]}'

**Expected:**

- Status 200  
- prediction: "Setosa"  
- confidence: 0.99

---

### Test 3 : Prédiction (Versicolor)

curl \-X POST http://localhost:5000/predict \\

  \-H "Content-Type: application/json" \\

  \-d '{"features": \[6.5, 2.8, 4.6, 1.5\]}'

**Expected:** prediction: "Versicolor"

---

### Test 4 : Prédiction (Virginica)

curl \-X POST http://localhost:5000/predict \\

  \-H "Content-Type: application/json" \\

  \-d '{"features": \[7.6, 3.0, 6.6, 2.2\]}'

**Expected:** prediction: "Virginica"

---

### Test 5 : Erreur de validation

curl \-X POST http://localhost:5000/predict \\

  \-H "Content-Type: application/json" \\

  \-d '{"features": \[5.1, 3.5\]}'

**Expected:**

- Status 400 Bad Request  
- error: "Nombre de features invalide"

---

### Tests avec Postman

Importer la collection Postman : `tests/postman_collection.json`

\# Ou lancer les tests automatiquement

pytest tests/test\_api.py \-v

---

## 🌐 Déploiement

### Déploiement sur AWS EC2

#### Étape 1 : Créer une instance EC2

\# AWS Console

1\. EC2 → Instances → Launch Instance

2\. AMI : Ubuntu 22.04 LTS

3\. Instance Type : t2.micro (free tier)

4\. Security Group : Ouvrir ports 22 (SSH) et 5000 (HTTP)

5\. Key Pair : Télécharger la clé .pem

#### Étape 2 : Connexion SSH

ssh \-i "your-key.pem" ubuntu@\<EC2\_PUBLIC\_IP\>

#### Étape 3 : Installation de Docker

\# Mettre à jour le système

sudo apt-get update && sudo apt-get upgrade \-y

\# Installer Docker

sudo apt-get install \-y docker.io

\# Ajouter l'utilisateur au groupe docker

sudo usermod \-aG docker $USER

newgrp docker

\# Vérifier l'installation

docker \--version

#### Étape 4 : Déployer le conteneur

\# Pull l'image depuis DockerHub

docker pull username/ml-api:latest

\# Lancer le conteneur

docker run \-d \\

  \-p 5000:5000 \\

  \--name ml-api-prod \\

  \--restart unless-stopped \\

  username/ml-api:latest

\# Vérifier

docker ps

#### Étape 5 : Tester en production

\# Depuis votre machine locale

curl http://\<EC2\_PUBLIC\_IP\>:5000/health

curl \-X POST http://\<EC2\_PUBLIC\_IP\>:5000/predict \\

  \-H "Content-Type: application/json" \\

  \-d '{"features": \[5.1, 3.5, 1.4, 0.2\]}'

---

## 📁 Structure du projet

devops-mlops-aws-student-project/

├── README.md                          \# Ce fichier

├── .gitignore                         \# Fichiers à ignorer dans Git

├── .dockerignore                      \# Fichiers à ignorer dans Docker

│

├── requirements.txt                   \# Dépendances Python

├── Dockerfile                         \# Configuration Docker

├── docker-compose.yml                 \# (Optionnel) Orchestration containers

│

├── api/                               \# Code de l'API Flask

│   ├── \_\_init\_\_.py                    \# Initialisation Flask

│   ├── app.py                         \# Application Flask principale

│   ├── model\_loader.py                \# Chargement des modèles ML

│   ├── routes.py                      \# Endpoints de l'API

│   └── config.py                      \# Configuration

│

├── model/                             \# Modèles ML sérialisés

│   ├── model.pkl                      \# Decision Tree entraîné

│   └── scaler.pkl                     \# StandardScaler pour normalisation

│

├── notebooks/                         \# Jupyter Notebooks

│   └── train\_model.ipynb              \# Notebook d'entraînement

│

├── tests/                             \# Tests unitaires

│   ├── test\_api.py                    \# Tests des endpoints

│   └── postman\_collection.json        \# Collection Postman

│

├── .github/                           \# GitHub Actions

│   └── workflows/

│       └── ci.yml                     \# Pipeline CI/CD

│

└── docker/                            \# (Optionnel) Configurations Docker supplémentaires

    ├── Dockerfile                     \# Alternative au Dockerfile racine

    └── .dockerignore                  \# Alternative au .dockerignore racine

---

## 🔄 Pipeline CI/CD (GitHub Actions)

Le pipeline s'active automatiquement lors d'un push sur `main` :

1\. Checkout Code

2\. Build Docker Image

3\. Test Image (health check)

4\. Login to DockerHub

5\. Push to DockerHub

6\. Status: ✓ SUCCESS

**Pour déclencher manuellement :**

git push origin main

\# Voir les logs : GitHub → Actions → Workflow runs

---

## 📊 Métriques de performance

| Métrique | Valeur |
| :---- | :---- |
| **Accuracy** | 96.67% |
| **Latence API** | \< 50 ms |
| **Image Size** | 425 MB |
| **Build Time** | \~2 minutes |
| **Deployment Time** | \< 5 minutes |
| **Memory Usage** | \~200 MB |
| **CPU Usage** | \< 10% (idle) |

---

## 🛠️ Commandes utiles

### Docker

\# Build

docker build \-t ml-api:1.0 .

\# Run

docker run \-d \-p 5000:5000 \--name ml-api ml-api:1.0

\# List containers

docker ps \-a

\# View logs

docker logs \-f ml-api

\# Stop container

docker stop ml-api

\# Remove container

docker rm ml-api

\# Remove image

docker rmi ml-api:1.0

\# Execute command in container

docker exec \-it ml-api /bin/bash

\# Inspect container

docker inspect ml-api

### Git

\# Clone repository

git clone \<repo-url\>

\# Create feature branch

git checkout \-b feature/my-feature

\# Commit changes

git add .

git commit \-m "feat: add new feature"

\# Push to GitHub

git push origin feature/my-feature

\# Create Pull Request

\# (via GitHub Web Interface  
