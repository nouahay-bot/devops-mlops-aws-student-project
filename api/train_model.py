import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Charger Iris
X, y = load_iris(return_X_y=True)

# Entraînement
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_scaled, y)

# Sauvegarder
joblib.dump(clf, os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

print("✅ Modèle et scaler sauvegardés avec random_state=42")
