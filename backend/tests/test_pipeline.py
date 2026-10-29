from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_and_components():
    assert client.get('/api/v1/health').status_code == 200
    items = client.get('/api/v1/components').json()['items']
    assert any(item['type'] == 'model.tune' for item in items)

def test_complete_pipeline_persists_and_emits_artifacts():
    response = client.post('/api/v1/pipelines/classification/run', json={'config': {'samples': 160, 'random_seed': 7, 'quality_threshold': 0.5, 'run_tuning': True}})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == 'succeeded'
    assert result['metrics']['selected']['f1'] >= 0
    assert result['comparison']['trials']
    assert result['quality_gate']['metric'] == 'f1'
    assert any(a['name'] == 'Evaluation report' for a in result['artifacts'])
    assert client.get(f"/api/v1/runs/{result['run_id']}").status_code == 200
