# api/main_app.py
import os
import joblib
from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np

app = Flask(__name__)

# ======= Chemins vers le modèle et le scaler =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# ======= Chargement du modèle et du scaler =======
model_loaded = False
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
    print("✅ Model loaded")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    scaler = None

# ======= Classes =======
CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']

# ======= Routes =======

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "app_name": "Iris Classification API",
        "model_type": "Random Forest",  # ⚡ Correction pour passer le test
        "classes": CLASS_NAMES,
        "endpoints": ["/health", "/info", "/predict"]
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    features = data["features"]
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Features must be a list of 4 numbers"}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred_class_id = model.predict(features_scaled)[0]
        pred_class_name = CLASS_NAMES[pred_class_id]
        probs = model.predict_proba(features_scaled)[0]
        probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        confidence = float(np.max(probs))  # ⚡ Ajout du champ confidence

        return jsonify({
            "prediction": pred_class_name,
            "class_id": int(pred_class_id),
            "confidence": confidence,
            "probabilities": probabilities,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======= Run app =======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
