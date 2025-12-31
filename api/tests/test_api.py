import pytest
from pathlib import Path
import sys

# Ajouter le répertoire parent au path pour que Python trouve api/

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
# Importer l'app Flask réelle

from api.main_app import app as flask_app

# ====================================================================
# FIXTURES
# ====================================================================

@pytest.fixture
def client():
    """Crée un client test pour faire des requêtes"""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

# ====================================================================
# TESTS : HEALTH CHECK
# ====================================================================

class TestHealth:
    """Tests du endpoint /health"""

    def test_health_check_status_200(self, client):
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_check_content(self, client):
        response = client.get('/health')
        data = response.get_json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'timestamp' in data
        assert data['status'] == 'healthy'

    def test_health_check_model_loaded(self, client):
        response = client.get('/health')
        data = response.get_json()
        assert data['model_loaded'] is True

    def test_health_check_content_type(self, client):
        response = client.get('/health')
        assert response.content_type == 'application/json'

# ====================================================================
# TESTS : PREDICT - CAS VALIDES
# ====================================================================

class TestPredictValid:
    """Tests du endpoint /predict avec données valides"""

    def test_predict_setosa(self, client):
        response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
        assert response.status_code == 200
        data = response.get_json()
        assert data['prediction'] == 'Setosa'
        assert data['class_id'] == 0
        assert 0 <= data['confidence'] <= 1
        assert set(data['probabilities'].keys()) == {'Setosa', 'Versicolor', 'Virginica'}
        assert 'timestamp' in data

    def test_predict_versicolor(self, client):
        response = client.post('/predict', json={'features': [6.5, 2.8, 4.6, 1.5]})
        assert response.status_code == 200
        data = response.get_json()
        assert data['prediction'] == 'Versicolor'
        assert data['class_id'] == 1

    def test_predict_virginica(self, client):
        response = client.post('/predict', json={'features': [7.6, 3.0, 6.6, 2.2]})
        assert response.status_code == 200
        data = response.get_json()
        assert data['prediction'] == 'Virginica'
        assert data['class_id'] == 2

    def test_predict_probabilities_sum_to_one(self, client):
        response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
        data = response.get_json()
        total = sum(data['probabilities'].values())
        assert abs(total - 1.0) < 0.001

# ====================================================================
# TESTS : PREDICT - CAS INVALIDES
# ====================================================================

class TestPredictInvalid:

    def test_missing_features(self, client):
        response = client.post('/predict', json={'data': [5.1, 3.5, 1.4, 0.2]})
        assert response.status_code == 400

    def test_empty_json(self, client):
        response = client.post('/predict', json={})
        assert response.status_code == 400

    def test_wrong_number_features(self, client):
        response = client.post('/predict', json={'features': [5.1, 3.5]})
        assert response.status_code == 400
        response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2, 0.5]})
        assert response.status_code == 400

    def test_non_numeric_features(self, client):
        response = client.post('/predict', json={'features': ['a', 'b', 'c', 'd']})
        assert response.status_code in [400, 500]

# ====================================================================
# TESTS : INFO ENDPOINT
# ====================================================================

class TestInfo:

    def test_info_status_200(self, client):
        response = client.get('/info')
        assert response.status_code == 200

    def test_info_content(self, client):
        response = client.get('/info')
        data = response.get_json()
        assert data['app_name'] == 'Iris Classification API'
        assert 'Random Forest' in data['model_type']
        assert data['classes'] == ['Setosa', 'Versicolor', 'Virginica']
        assert 'endpoints' in data

# ====================================================================
# TESTS : ERROR HANDLING
# ====================================================================

class TestErrorHandling:

    def test_404_not_found(self, client):
        response = client.get('/nonexistent')
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        response = client.get('/predict')  # GET sur /predict (POST requis)
        assert response.status_code == 405

# ====================================================================
# TESTS : PERFORMANCE
# ====================================================================

class TestPerformance:

    def test_health_check_response_time(self, client):
        import time
        start = time.time()
        response = client.get('/health')
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 0.5

    def test_predict_response_time(self, client):
        import time
        start = time.time()
        response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 1.0

# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
