# main_app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Paths vers les fichiers modèles et scaler
MODEL_PATH = os.path.join("api", "models", "model.pkl")
SCALER_PATH = os.path.join("api", "models", "scaler.pkl")

# Chargement du modèle et du scaler
model_loaded = False
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Endpoint health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

# Endpoint info
@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "app_name": "Iris Classification API",
        "version": "1.0.0"
    })

# Endpoint predict
@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in JSON"}), 400

    features = data["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Features must be a list of length 4"}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        return jsonify({
            "prediction": str(prediction),
            "probabilities": {
                "setosa": float(probabilities[0]),
                "versicolor": float(probabilities[1]),
                "virginica": float(probabilities[2])
            }
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# Error handling
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    import os
    workers = int(os.environ.get("GUNICORN_WORKERS", 2))
    threads = int(os.environ.get("GUNICORN_THREADS", 2))
    from gunicorn.app.base import BaseApplication

    class FlaskApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        "bind": "0.0.0.0:5000",
        "workers": workers,
        "threads": threads
    }
    FlaskApplication(app, options).run()
