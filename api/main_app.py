"""
api/main_app.py
Application Flask avec tous les endpoints
"""

from flask import Flask, request, jsonify
from datetime import datetime
import logging
import os

from .model_loader import model_loader  # import relatif correct
model_loader.load_models()
# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Créer l'application Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # 1MB max

logger.info("=" * 60)
logger.info("Initialisation de l'application Flask")
logger.info("=" * 60)

# Charger les modèles
if model_loader.load_models():
    logger.info("Model ML chargé avec succès")
else:
    logger.warning("Erreur lors du chargement des modèles ML")


# ============================================================================
# ENDPOINT 1 : Health Check
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé - Vérifie l'état du service"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader.is_loaded,
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# ENDPOINT 2 : Prédiction
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        if not model_loader.is_loaded:
            return jsonify({
                'error': 'Modèle non disponible',
                'message': 'Le modèle ML n\'a pas pu être chargé'
            }), 503

        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Paramètre manquant',
                'message': 'Le JSON doit contenir la clé "features"'
            }), 400

        features = data['features']
        if not isinstance(features, list) or len(features) != 4:
            return jsonify({
                'error': 'Nombre de features invalide',
                'message': f'Attendu 4 features, reçu {len(features) if isinstance(features, list) else "N/A"}'
            }), 400

        prediction, probs, confidence = model_loader.predict(features)
        if prediction is None:
            return jsonify({'error': 'Erreur lors de la prédiction'}), 500

        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        class_id = iris_classes.index(prediction)

        response = {
            'prediction': prediction,
            'class_id': class_id,
            'probabilities': probs,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"Prédiction réussie : {prediction} (confiance: {confidence:.2%})")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({
            'error': 'Erreur serveur',
            'message': str(e)
        }), 500


# ============================================================================
# ENDPOINT 3 : Informations API
# ============================================================================
@app.route('/info', methods=['GET'])
def info():
    """Endpoint d'informations sur l'API"""
    return jsonify({
        'app_name': 'Iris Classification API',
        'version': '1.0.0',
        'model_type': 'Random Forest Classifier',
        'dataset': 'Iris Dataset',
        'classes': ['Setosa', 'Versicolor', 'Virginica'],
        'num_features': 4,
        'feature_names': [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ],
        'model_info': {
            'accuracy': 0.9667,
            'precision': 0.9697,
            'recall': 0.9667,
            'f1_score': 0.9666
        },
        'endpoints': {
            'GET /health': 'Vérifier l\'état du service',
            'POST /predict': 'Faire une prédiction',
            'GET /info': 'Informations sur l\'API'
        }
    }), 200


# ============================================================================
# Error Handlers
# ============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur serveur : {error}")
    return jsonify({'error': 'Internal server error'}), 500


logger.info("Application Flask initialisée avec succès")
logger.info("=" * 60)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
