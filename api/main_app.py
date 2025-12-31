from flask import Flask, request, jsonify
from pathlib import Path
import joblib
from datetime import datetime
import numpy as np

app = Flask(__name__)

# =====================
# CHARGEMENT DU MODELE
# =====================
MODEL_PATH = Path(__file__).parent / "model/iris_rf_model.joblib"

model_loaded = False
model_data = {}

try:
    if MODEL_PATH.exists():
        model_data = joblib.load(MODEL_PATH)
        model_loaded = True
    else:
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

model = model_data.get("model")
scaler = model_data.get("scaler")
class_names = ['Setosa', 'Versicolor', 'Virginica']

# =====================
# ENDPOINT HEALTH
# =====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat()
    })

# =====================
# ENDPOINT INFO
# =====================
@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "app_name": "Iris Classification API",
        "model_type": type(model).__name__ if model else "Unknown",
        "classes": class_names,
        "endpoints": ["/health", "/predict", "/info"]
    })

# =====================
# ENDPOINT PREDICT
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = data["features"]
    if len(features) != 4:
        return jsonify({"error": "Wrong number of features"}), 400

    try:
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]
        class_id = int(np.argmax(probs))
        prediction = class_names[class_id]
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        confidence = float(probs[class_id])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        "prediction": prediction,
        "class_id": class_id,
        "confidence": confidence,
        "probabilities": prob_dict,
        "timestamp": datetime.utcnow().isoformat()
    })

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
