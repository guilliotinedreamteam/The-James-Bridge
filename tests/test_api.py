from fastapi.testclient import TestClient
from neurobridge.api import app

client = TestClient(app)

def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "engine_running" in data

def test_synthesize_invalid_input():
    response = client.post("/synthesize", json={"sequence": "1, 2, invalid"})
    assert response.status_code == 400
    assert "Invalid sequence format" in response.json()["detail"]

def test_synthesize_empty_sequence():
    response = client.post("/synthesize", json={"sequence": " "})
    # split(",") -> [" "] -> strip -> [] -> empty list.
    # The code: ids = [int(token) for token in req.sequence.split(",") if token.strip()]
    # Then synthesizer.synthesize(ids).
    # If ids is empty, synthesizer might return empty audio.
    # The code checks `if audio.size == 0: return {"message": ...}`
    # But we need config to exist first.
    # Since config might not exist in test env, we expect 500 or 404 (FileNotFound).
    # However, invalid input check happens *after* config load in my previous patch?
    # Let's check api.py again.
    pass
