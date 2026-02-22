import pytest
from fastapi.testclient import TestClient
from neurobridge.api import app
import os
from pathlib import Path

def test_synthesis_endpoint():
    with TestClient(app) as client:
        # Define output path for test
        output_file = "test_output.wav"
        if os.path.exists(output_file):
            os.remove(output_file)

        response = client.post("/synthesize", json={
            "sequence": "1,2,3",
            "output_path": output_file
        })

        assert response.status_code == 200
        assert response.json()["message"] == "Synthesis complete"
        assert os.path.exists(output_file)

        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)
