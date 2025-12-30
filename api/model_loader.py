"""
api/model_loader.py
Gère le chargement et l'utilisation des modèles ML
"""

import joblib
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Chemins des fichiers modèles
MODEL_DIR = Path(__file__).parent.parent / "/app/model"
MODEL_PATH = MODEL_DIR /"/app/model/model.pkl"
SCALER_PATH = MODEL_DIR / "/app/model/scaler.pkl"

class ModelLoader:
    """Gère le chargement et la prédiction des modèles ML (Singleton)"""

    _instance = None

    def __new__(cls):
        """Implémente le singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialise les attributs"""
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.error_message = None

    def load_models(self) -> bool:
        """
        Charge les modèles ML

        Returns:
            bool: True si succès, False sinon
        """
        try:
            # Vérifier que les fichiers existent
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Modèle non trouvé : {MODEL_PATH}")

            if not SCALER_PATH.exists():
                raise FileNotFoundError(f"Scaler non trouvé : {SCALER_PATH}")

            # Charger le modèle
            logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            logger.info("✓ Modèle chargé avec succès")

            # Charger le scaler
            logger.info(f"Chargement du scaler depuis {SCALER_PATH}")
            self.scaler = joblib.load(SCALER_PATH)
            logger.info("✓ Scaler chargé avec succès")

            self.is_loaded = True
            self.error_message = None

            return True

        except FileNotFoundError as e:
            error_msg = f"Fichier modèle non trouvé : {str(e)}"
            logger.error(error_msg)
            self.error_message = error_msg
            self.is_loaded = False
            return False

        except Exception as e:
            error_msg = f"Erreur lors du chargement des modèles : {str(e)}"
            logger.error(error_msg)
            self.error_message = error_msg
            self.is_loaded = False
            return False

    def predict(self, features: list) -> Tuple[Optional[str], Optional[dict], Optional[float]]:
        """
        Fait une prédiction sur les features données

        Args:
            features : Liste de 4 nombres (features Iris)

        Returns:
            Tuple : (prediction_class, probabilities_dict, confidence)
                    ou (None, None, None) si erreur
        """
        try:
            # Vérifier que les modèles sont chargés
            if not self.is_loaded:
                logger.error("Modèles non chargés")
                return None, None, None

            # Validation des features
            if not isinstance(features, list) or len(features) != 4:
                logger.error(f"Features invalides : {features}")
                return None, None, None

            # Conversion en numpy array
            features_array = np.array(features, dtype=float).reshape(1, -1)

            # Normalisation avec le scaler
            features_scaled = self.scaler.transform(features_array)

            # Prédiction
            prediction_class = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities))

            # Formatage des probabilités
            iris_classes = ['Setosa', 'Versicolor', 'Virginica']
            probs_dict = {
                iris_classes[i]: float(probabilities[i])
                for i in range(len(iris_classes))
            }

            prediction_name = iris_classes[prediction_class]

            logger.info(f"Prédiction : {prediction_name} (confiance: {confidence:.2%})")

            return prediction_name, probs_dict, confidence

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction : {str(e)}")
            return None, None, None

    def get_model_info(self) -> dict:
        """
        Retourne les informations sur les modèles chargés

        Returns:
            dict : Informations sur le modèle
        """
        return {
            'model_loaded': self.is_loaded,
            'model_type': type(self.model).__name__ if self.model else None,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'error_message': self.error_message,
            'model_path': str(MODEL_PATH),
            'scaler_path': str(SCALER_PATH)
        }


# Instance globale singleton
model_loader = ModelLoader()