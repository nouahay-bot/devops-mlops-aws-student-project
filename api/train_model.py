import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# 1️⃣ Racine du projet PyCharm (là où se trouve 'api' et 'models')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # si ce script est dans 'api'
PROJECT_ROOT = os.path.join(PROJECT_ROOT, '..')            # remonte d'un niveau
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)              # chemin absolu

# 2️⃣ Chemin vers le dossier models
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)  # crée le dossier s'il n'existe pas

# 3️⃣ Charger les données
X, y = load_iris(return_X_y=True)

# 4️⃣ Entraîner le modèle
clf = RandomForestClassifier()
clf.fit(X, y)

# 5️⃣ Sauvegarder le modèle et le scaler
model_path = os.path.join(MODEL_DIR, 'model.pkl')
joblib.dump(clf, model_path)

scaler = StandardScaler()
scaler.fit(X)
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

print(f"✅ Modèle sauvegardé dans {model_path}")
print(f"✅ Scaler sauvegardé dans {scaler_path}")

