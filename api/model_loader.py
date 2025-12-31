# api/model_loader.py

import os
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False

    def load_models(self):
        """Charge model.pkl et scaler.pkl depuis le dossier models"""
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            model_dir = os.path.join(project_root, 'models')

            model_path = os.path.join(model_dir, 'model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning("⚠️ Fichiers de modèles non trouvés dans %s", model_dir)
                return False

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
            logger.info("✓ Modèles chargés depuis %s", model_dir)
            return True

        except Exception as e:
            logger.warning("Erreur lors du chargement du modèle : %s", e)
            return False

    def predict(self, features):
        """Fait une prédiction et renvoie (classe, probas, confiance)"""
        if not self.is_loaded:
            return None, None, None

        try:
            x_scaled = self.scaler.transform([features])
            probs = self.model.predict_proba(x_scaled)[0]
            classes = ['Setosa', 'Versicolor', 'Virginica']
            pred_idx = int(np.argmax(probs))
            prediction = classes[pred_idx]
            confidence = float(probs[pred_idx])
            return prediction, dict(zip(classes, probs)), confidence
        except Exception as e:
            logger.error("Erreur prediction: %s", e)
            return None, None, None

# Créer l'instance unique que main_app importera
model_loader = ModelLoader()
