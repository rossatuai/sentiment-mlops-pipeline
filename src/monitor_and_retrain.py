from evaluate import evaluate_model
import subprocess
import mlflow

THRESHOLD = 0.99

# Use same MLflow location as training
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("model-monitoring")

print("Starting model performance monitoring...")


# Run evaluation
accuracy = evaluate_model()

print(f"Current Model Accuracy: {accuracy:.2f}")


# Log metrics
with mlflow.start_run():

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Threshold monitoring
    if accuracy < THRESHOLD:

        print("Model performance degraded.")
        print("Triggering automatic retraining...")

        mlflow.log_param(
            "retraining_triggered",
            True
        )

        subprocess.run(
            ["python", "src/train.py"]
        )

    else:

        print("Model performance acceptable.")

        mlflow.log_param(
            "retraining_triggered",
            False
        )