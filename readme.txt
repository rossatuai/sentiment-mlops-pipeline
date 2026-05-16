
# Start MLflow UI 
python -m mlflow ui --port 5001



# Sentiment Analysis MLOps Pipeline

An end-to-end MLOps pipeline for sentiment analysis built using Python, GitHub Actions, Docker, MLflow, Kubernetes (Kind), and ArgoCD.

This project demonstrates how machine learning models can move from development to production through automated workflows including training, testing, containerization, deployment, continuous integration, continuous delivery, and continuous retraining.

---

## Project Overview

The goal of this project is to demonstrate MLOps principles by creating a complete automated machine learning workflow rather than focusing only on model development.

The system performs sentiment classification on text inputs and automatically manages the machine learning lifecycle.

Features include:

- Data preprocessing
- Model training and testing
- MLflow experiment tracking
- Flask API deployment
- Docker containerization
- GitHub Actions automation
- Continuous Integration (CI)
- Continuous Delivery (CD)
- Continuous Training (CT)
- Kubernetes deployment using Kind
- GitOps deployment using ArgoCD

---

## Project Architecture

```text
User Input
     ↓
Flask API
     ↓
Trained Sentiment Model
     ↓
Prediction Returned

Development Workflow:

Code Push
     ↓
GitHub Actions
     ↓
Run Tests
     ↓
Train/Retrain Model
     ↓
Log Results to MLflow
     ↓
Build Docker Image
     ↓
Push Image to DockerHub
     ↓
Update Kubernetes Manifest
     ↓
ArgoCD Detects Change
     ↓
Sync to Kind Cluster
     ↓
Application Updated
```

---

## Tech Stack

### Machine Learning

- Python
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

### MLOps

- MLflow
- GitHub Actions
- Docker
- DockerHub
- Kubernetes (Kind)
- ArgoCD

### API

- Flask

### Testing

- Pytest

---

## Repository Structure

```text
sentiment-mlops-pipeline/

│
├── .github/
│   └── workflows/
│       ├── train.yml
│       ├── deploy.yml
│       └── retrain.yml
│
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
│
├── tests/
│   ├── test_model.py
│   └── test_api.py
│
├── app.py
├── train.py
├── retrain.py
├── Dockerfile
├── requirements.txt
├── mlruns/
├── models/
└── README.md
```

---

## MLOps Pipeline Stages

### 1. Data Acquisition & Preprocessing

The dataset is loaded and cleaned before being transformed using TF-IDF vectorization.

Steps:

- Load sentiment dataset
- Clean text
- Remove invalid values
- Split into train/test datasets
- Transform text into numerical features

---

### 2. Model Training & Testing

A Logistic Regression model is trained on processed text data.

Model outputs:

- Sentiment prediction
- Accuracy
- F1 score

The project focuses on operationalization rather than model experimentation.

---

### 3. MLflow Experiment Tracking

MLflow tracks:

- Accuracy metrics
- F1 score
- Hyperparameters
- Model artifacts

This allows comparison between model runs and enables reproducibility.

To start MLflow locally:

```bash
mlflow ui
```

Open:

```bash
http://localhost:5000
```

---

## Continuous Integration (CI)

GitHub Actions automatically runs when code is pushed.

CI workflow:

- Install dependencies
- Run unit tests
- Validate project files
- Train model
- Log results to MLflow

Benefits:

- Prevents broken code entering production
- Automatically validates updates

---

## Continuous Delivery (CD)

After successful testing:

1. Docker image is built
2. Image pushed to DockerHub
3. Kubernetes deployment file updated
4. ArgoCD detects changes
5. Cluster automatically updates

---

## Continuous Training (CT)

The project supports automatic retraining.

Retraining is triggered when:

- Model performance drops below a threshold
- Dataset changes
- Configuration changes

Retraining workflow:

```text
Performance Threshold Trigger
            ↓
GitHub Actions Retraining Workflow
            ↓
Train New Model
            ↓
Log to MLflow
            ↓
Build New Docker Image
            ↓
Deploy Updated Model
```

---

## Docker Deployment

Build image:

```bash
docker build -t sentiment-api .
```

Run locally:

```bash
docker run -p 5000:5000 sentiment-api
```

Application:

```bash
http://localhost:5000
```

---

## Kubernetes Deployment (Kind)

Create cluster:

```bash
kind create cluster --name sentiment-cluster
```

Apply manifests:

```bash
kubectl apply -f kubernetes/
```

Check deployment:

```bash
kubectl get all
```

---

## GitOps Deployment with ArgoCD

ArgoCD continuously watches the Git repository.

When GitHub Actions updates deployment files:

- ArgoCD detects changes
- Synchronizes Kubernetes automatically
- Updates running containers

Advantages:

- Automated deployments
- Reduced manual intervention
- Git becomes source of truth
- Easy rollback capability

---

## Branching Strategy

This project uses a simple Git workflow:

Main branch:

- Stable production code

Feature branches:

- Development work
- Testing changes before merge

Workflow:

```text
Feature Branch
      ↓
Pull Request
      ↓
GitHub Actions Tests
      ↓
Merge to Main
      ↓
Deployment Triggered
```

---

## Running Locally

Clone repository:

```bash
git clone https://github.com/rossatuai/sentiment-mlops-pipeline.git
```

Move into project:

```bash
cd sentiment-mlops-pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python train.py
```

Run API:

```bash
python app.py
```

---

## Example API Request

Input:

```json
{
    "text":"This product is amazing"
}
```

Output:

```json
{
    "prediction":"Positive"
}
```

---

## Future Improvements

Potential enhancements:

- Prometheus monitoring
- Grafana dashboards
- Drift detection
- Multiple model comparisons
- Cloud deployment (AWS/GCP)
- Automatic rollback on failed deployment

---

## Author

Ross M

Big Data Analytics – MLOps Assessment Project

---

## License

Educational use only.