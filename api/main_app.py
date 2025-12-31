# api/main_app.py
from flask import Flask, jsonify, request
from datetime import datetime
import pickle
import numpy as np
import os

app = Flask(__name__)

# ---------------------------
# Variables globales
# ---------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

model = None
scaler = None
classes = ['Setosa', 'Versicolor', 'Virginica']

# ---------------------------
# Chargement du modèle
# ---------------------------
def load_model():
    global model, scaler
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        scaler = None

load_model()

# ---------------------------
# Endpoints
# ---------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    features = data.get("features")
    if features is None:
        return jsonify({"error": "Missing 'features' key"}), 400

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Invalid number of features"}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        probs = model.predict_proba(features_scaled)[0]
        class_idx = int(model.predict(features_scaled)[0])
        prediction = classes[class_idx]
        return jsonify({
            "prediction": prediction,
            "class_id": class_idx,
            "confidence": float(max(probs)),
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "app_name": "Iris Classification API",
        "model_type": "Random Forest",
        "classes": classes,
        "endpoints": ["/health", "/predict", "/info"]
    }), 200

# ---------------------------
# Gestion des erreurs
# ---------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
