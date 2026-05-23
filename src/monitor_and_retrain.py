# Import model evaluation function
from evaluate import evaluate_model

# Used to run training script automatically
import subprocess

# Import MLflow for tracking metrics
import mlflow

# Accuracy threshold for retraining
THRESHOLD = 0.80

# Use same MLflow location as training
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("model-monitoring")

print("Starting model performance monitoring...")


# Run model evaluation
accuracy = evaluate_model()

print(f"Current Model Accuracy: {accuracy:.2f}")


# Start MLflow tracking run
with mlflow.start_run():

    # Save accuracy metric to MLflow
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Check if accuracy drops below threshold
    if accuracy < THRESHOLD:

        print("Model performance degraded.")
        print("Triggering automatic retraining...")

        # Record retraining event
        mlflow.log_param(
            "retraining_triggered",
            True
        )

        # Run training script again
        subprocess.run(
            ["python", "src/train.py"]
        )

    else:

        print("Model performance acceptable.")

        # Record that retraining was not needed
        mlflow.log_param(
            "retraining_triggered",
            False
        )