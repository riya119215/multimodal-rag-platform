import pytest
from fastapi.testclient import TestClient
from app.main import api_app

client = TestClient(api_app)

def test_api_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "embedding_model" in data

def test_api_documents_endpoint():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_api_code_execution_endpoint():
    code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
print('Plot generated successfully')
"""
    response = client.post("/api/v1/execute-code", json={"code": code})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["has_plot"] is True
    assert "Plot generated successfully" in data["output"]
