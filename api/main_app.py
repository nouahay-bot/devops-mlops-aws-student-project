from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# -------------------------
# Load model and scaler
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

model_loaded = False
model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# Mapping numeric labels to class names
CLASS_MAPPING = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# -------------------------
# Health endpoint
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    from datetime import datetime
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat()
    })

# -------------------------
# Predict endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    features = data.get("features")

    if features is None or len(features) != 4:
        return jsonify({"error": "Invalid input"}), 400

    try:
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction_num = model.predict(features_scaled)[0]
        prediction = CLASS_MAPPING.get(prediction_num, str(prediction_num))
        probas = model.predict_proba(features_scaled)[0]
        probabilities = {name: float(probas[i]) for i, name in CLASS_MAPPING.items()}

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Info endpoint
# -------------------------
@app.route("/info", methods=["GET"])
def info():
    model_type = type(model).__name__ if model_loaded else "Unknown"
    return jsonify({
        "app_name": "Iris Classification API",
        "version": "1.0.0",
        "model_type": model_type
    })

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

