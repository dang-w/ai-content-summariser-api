from fastapi.testclient import TestClient
import sys
import os

# Import the app from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def test_summarise_endpoint():
    response = client.post(
        "/api/summarise",
        json={
            "text": "This is a test paragraph that should be summarized.",
            "max_length": 50,
            "min_length": 10
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "original_text_length" in data
    assert "summary_length" in data
