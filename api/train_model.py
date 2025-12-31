import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# 1️⃣ Chemin vers le dossier models
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # script dans 'api'
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 2️⃣ Charger les données
X, y = load_iris(return_X_y=True)

# 3️⃣ Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Entraînement RandomForest
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# 5️⃣ Sauvegarde
joblib.dump(clf, os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

print("✅ Modèle et scaler sauvegardés dans", MODEL_DIR)
