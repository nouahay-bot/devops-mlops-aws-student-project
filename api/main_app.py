from flask import Flask, jsonify, request
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# =====================================================
# CONFIGURATION DU MODELE
# =====================================================
MODEL_PATH = os.path.join("api", "model", "model.pkl")
SCALER_PATH = os.path.join("api", "model", "scaler.pkl")
CLASSES = ["Setosa", "Versicolor", "Virginica"]

model = None
scaler = None
app.config['MODEL_LOADED'] = False

def load_model_and_scaler():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        app.config['MODEL_LOADED'] = True
        print("Modèle et scaler chargés avec succès.")
    except Exception as e:
        print("Erreur chargement modèle/scaler:", e)
        app.config['MODEL_LOADED'] = False

load_model_and_scaler()

# =====================================================
# ENDPOINT /health
# =====================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": app.config.get('MODEL_LOADED', False),
        "timestamp": datetime.utcnow().isoformat()
    })

# =====================================================
# ENDPOINT /info
# =====================================================
@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "app_name": "Iris Classification API",
        "model_type": type(model).__name__ if model else "Unknown",
        "classes": CLASSES,
        "endpoints": ["/health", "/predict", "/info"]
    })

# =====================================================
# ENDPOINT /predict
# =====================================================
@app.route('/predict', methods=['POST'])
def predict():
    if not app.config.get('MODEL_LOADED', False):
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    features = data['features']
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Features must be a list of 4 numeric values"}), 400

    try:
        features_array = np.array([features], dtype=float)
        features_scaled = scaler.transform(features_array)
    except Exception:
        return jsonify({"error": "Features must be numeric"}), 400

    # Prédiction
    pred_class_id = int(model.predict(features_scaled)[0])
    pred_class_name = CLASSES[pred_class_id]
    pred_proba = model.predict_proba(features_scaled)[0]
    probabilities = {CLASSES[i]: float(pred_proba[i]) for i in range(len(CLASSES))}
    confidence = float(np.max(pred_proba))

    return jsonify({
        "prediction": pred_class_name,
        "class_id": pred_class_id,
        "confidence": confidence,
        "probabilities": probabilities,
        "timestamp": datetime.utcnow().isoformat()
    })

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
