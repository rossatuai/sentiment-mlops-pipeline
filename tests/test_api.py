# Import required libraries
import sys
import os

# Add project folder to system path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

# Import Flask application
from app.app import app


# Test home page route
def test_home_route():

    client = app.test_client()

    response = client.get("/")

    # Check response was successful
    assert response.status_code == 200


# Test invalid route
def test_invalid_route():

    client = app.test_client()

    response = client.get("/fake")

    # Check invalid page returns error
    assert response.status_code == 404


# Test prediction endpoint
def test_prediction_route():

    client = app.test_client()

    response = client.post(
        "/predict",
        json={"text": "I love this movie"}
    )

    # Check response was successful
    assert response.status_code == 200

    data = response.get_json()

    # Check prediction exists in response
    assert "prediction" in data