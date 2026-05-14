from evaluate import evaluate_model
import subprocess

THRESHOLD = 0.80

accuracy = evaluate_model()

print(f"Current Model Accuracy: {accuracy:.2f}")

if accuracy < THRESHOLD:

    print("Accuracy below threshold.")
    print("Retraining model...")

    subprocess.run(["python", "src/train.py"])

else:

    print("Model performance acceptable.")