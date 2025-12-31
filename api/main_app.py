# api/main_app.py
from flask import Flask, request, jsonify
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# =========================
# Chemins vers le modèle
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# =========================
# Chargement du modèle et du scaler
# =========================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Erreur chargement modèle ou scaler : {e}")
    model = None
    scaler = None
    MODEL_LOADED = False

CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']

# =========================
# Endpoints
# =========================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "app_name": "Iris Classification API",
        "model_type": "Random Forest",
        "classes": CLASS_NAMES,
        "endpoints": ["/predict", "/health", "/info"]
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    features = data.get('features')

    # Validation simple
    if not features or len(features) != 4:
        return jsonify({"error": "Invalid input, expected 4 features"}), 400

    try:
        # Transformation avec scaler
        features_scaled = scaler.transform([features])
        class_id = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]

        return jsonify({
            "prediction": CLASS_NAMES[class_id],
            "class_id": class_id,
            "confidence": float(max(probabilities)),
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Run
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
