import pytest
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importer l'app
try:
    from main_app import app
except ImportError:
    # For IDE resolution only
    from flask import Flask
    app = Flask(__name__)  # Dummy app for IDE
    print("Warning: Using dummy app for IDE resolution")


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """
    Créer un client test pour faire des requêtes
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ============================================================================
# TESTS : HEALTH CHECK
# ============================================================================

class TestHealth:
    """Tests du endpoint /health"""

    def test_health_check_status_200(self, client):
        """Le health check doit retourner 200"""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_check_content(self, client):
        """Le health check doit contenir les bonnes données"""
        response = client.get('/health')
        data = response.get_json()

        assert 'status' in data
        assert 'model_loaded' in data
        assert 'timestamp' in data
        assert data['status'] == 'healthy'

    def test_health_check_model_loaded(self, client):
        """Vérifier que le modèle est chargé"""
        response = client.get('/health')
        data = response.get_json()
        assert data['model_loaded'] is True

    def test_health_check_content_type(self, client):
        """Le health check doit retourner du JSON"""
        response = client.get('/health')
        assert response.content_type == 'application/json'


# ============================================================================
# TESTS : PREDICT - CAS VALIDES
# ============================================================================

class TestPredictValid:
    """Tests du endpoint /predict avec données valides"""

    def test_predict_setosa(self, client):
        """Test prédiction d'une Setosa"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2]
        })

        assert response.status_code == 200
        data = response.get_json()

        assert 'prediction' in data
        assert 'class_id' in data
        assert 'probabilities' in data
        assert 'confidence' in data

        assert data['prediction'] == 'Setosa'
        assert data['class_id'] == 0
        assert isinstance(data['confidence'], float)
        assert 0 <= data['confidence'] <= 1

    def test_predict_versicolor(self, client):
        """Test prédiction d'une Versicolor"""
        response = client.post('/predict', json={
            'features': [6.5, 2.8, 4.6, 1.5]
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['prediction'] == 'Versicolor'
        assert data['class_id'] == 1

    def test_predict_virginica(self, client):
        """Test prédiction d'une Virginica"""
        response = client.post('/predict', json={
            'features': [7.6, 3.0, 6.6, 2.2]
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['prediction'] == 'Virginica'
        assert data['class_id'] == 2

    def test_predict_probabilities_sum_to_one(self, client):
        """Les probabilités doivent sommer à 1"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2]
        })

        data = response.get_json()
        probabilities = data['probabilities']

        # Calculer la somme
        total = sum(probabilities.values())

        # Vérifier que c'est proche de 1 (avec tolérance float)
        assert abs(total - 1.0) < 0.001

    def test_predict_all_classes_in_probabilities(self, client):
        """Les 3 classes doivent être dans les probabilités"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2]
        })

        data = response.get_json()
        probabilities = data['probabilities']

        assert 'Setosa' in probabilities
        assert 'Versicolor' in probabilities
        assert 'Virginica' in probabilities

    def test_predict_confidence_matches_max_probability(self, client):
        """La confiance doit être la probabilité max"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2]
        })

        data = response.get_json()
        probabilities = data['probabilities']
        confidence = data['confidence']

        max_prob = max(probabilities.values())
        assert abs(confidence - max_prob) < 0.001

    def test_predict_timestamp_present(self, client):
        """Un timestamp doit être dans la réponse"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2]
        })

        data = response.get_json()
        assert 'timestamp' in data
        assert isinstance(data['timestamp'], str)


# ============================================================================
# TESTS : PREDICT - CAS INVALIDES (ERREURS)
# ============================================================================

class TestPredictInvalid:
    """Tests du endpoint /predict avec données invalides"""

    def test_predict_missing_features_key(self, client):
        """Erreur si clé 'features' manquante"""
        response = client.post('/predict', json={
            'data': [5.1, 3.5, 1.4, 0.2]
        })

        assert response.status_code == 400

    def test_predict_empty_json(self, client):
        """Erreur si JSON vide"""
        response = client.post('/predict', json={})

        assert response.status_code == 400

    def test_predict_wrong_number_of_features(self, client):
        """Erreur si nombre de features ≠ 4"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5]
        })

        assert response.status_code == 400

    def test_predict_too_many_features(self, client):
        """Erreur si trop de features"""
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2, 0.5]
        })

        assert response.status_code == 400

    def test_predict_non_numeric_features(self, client):
        """Erreur si features ne sont pas numériques"""
        response = client.post('/predict', json={
            'features': ['a', 'b', 'c', 'd']
        })

        # Peut être 400 ou 500
        assert response.status_code in [400, 500]


# ============================================================================
# TESTS : INFO ENDPOINT
# ============================================================================

class TestInfo:
    """Tests du endpoint /info"""

    def test_info_status_200(self, client):
        """Info doit retourner 200"""
        response = client.get('/info')
        assert response.status_code == 200

    def test_info_contains_app_name(self, client):
        """Info doit contenir le nom de l'app"""
        response = client.get('/info')
        data = response.get_json()

        assert 'app_name' in data
        assert data['app_name'] == 'Iris Classification API'

    def test_info_contains_version(self, client):
        """Info doit contenir la version"""
        response = client.get('/info')
        data = response.get_json()

        assert 'version' in data

    def test_info_contains_model_type(self, client):
        """Info doit contenir le type de modèle"""
        response = client.get('/info')
        data = response.get_json()

        assert 'model_type' in data
        assert 'Decision Tree' in data['model_type']

    def test_info_contains_classes(self, client):
        """Info doit contenir les classes"""
        response = client.get('/info')
        data = response.get_json()

        assert 'classes' in data
        assert len(data['classes']) == 3
        assert 'Setosa' in data['classes']
        assert 'Versicolor' in data['classes']
        assert 'Virginica' in data['classes']

    def test_info_contains_endpoints(self, client):
        """Info doit lister les endpoints"""
        response = client.get('/info')
        data = response.get_json()

        assert 'endpoints' in data


# ============================================================================
# TESTS : ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Tests de la gestion des erreurs"""

    def test_404_not_found(self, client):
        """Route inexistante retourne 404"""
        response = client.get('/nonexistent')
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Mauvaise méthode HTTP retourne 405"""
        response = client.get('/predict')  # GET sur /predict (POST requis)
        assert response.status_code == 405


# ============================================================================
# TESTS : PERFORMANCE
# ============================================================================

class TestPerformance:
    """Tests de performance"""

    def test_health_check_response_time(self, client):
        """Le health check doit être très rapide"""
        import time

        start = time.time()
        response = client.get('/health')
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.5  # Moins de 500ms

    def test_predict_response_time(self, client):
        """La prédiction doit être rapide"""
        import time

        start = time.time()
        response = client.post('/predict', json={
            'features': [5.1, 3.5, 1.4, 0.2]
        })
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0  # Moins de 1 seconde


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    """Lancer les tests avec pytest"""
    pytest.main([__file__, '-v', '--tb=short'])