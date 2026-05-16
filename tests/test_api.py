import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app


def test_home_route():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_invalid_route():
    client = app.test_client()

    response = client.get("/fake")

    assert response.status_code == 404




def test_prediction_route():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={"text": "I love this movie"}
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data