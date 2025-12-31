import os
from flask import Flask, jsonify, request
from joblib import load

app = Flask(__name__)

# Chemins absolus du modèle et du scaler
MODEL_PATH = os.path.join(os.getcwd(), "model/model.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "model/scaler.pkl")

# Variables globales
model_loaded = False
model = None
scaler = None

# Chargement du modèle au démarrage
try:
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    model_loaded = True
    print("Modèle et scaler chargés avec succès.")
except Exception as e:
    model_loaded = False
    print("Erreur lors du chargement du modèle:", e)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = data["features"]
    if len(features) != 4:
        return jsonify({"error": "Invalid number of features"}), 400

    try:
        X_scaled = scaler.transform([features])
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

        return jsonify({
            "prediction": prediction,
            "probabilities": prob_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
