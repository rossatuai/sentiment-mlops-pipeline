# Sentiment Analysis MLOps Pipeline

End-to-end MLOps pipeline for sentiment analysis demonstrating automation across the complete machine learning lifecycle using GitHub Actions, MLflow, Docker, Kubernetes (Kind), and ArgoCD.

The project focuses on MLOps practices, including Continuous Integration (CI), Continuous Deployment (CD), Continuous Training (CT), experiment tracking, deployment automation, and model monitoring.

---

# Project Overview

This project demonstrates how machine learning systems move from development into production through automated workflows.

The system performs sentiment classification on text reviews while automating:

- Data preprocessing
- Model training
- Model evaluation
- MLflow experiment tracking
- Unit testing
- Docker image creation
- Continuous deployment
- Model monitoring
- Automatic retraining
- Kubernetes deployment
- GitOps synchronization

---

# Final Architecture Workflow


Developer
    ↓
GitHub Repository
    ↓
GitHub Actions
    ↓
Unit Tests (Pytest)
    ↓
Data Preprocessing
    ↓
Model Training
    ↓
MLflow Experiment Tracking
    ↓
Model Evaluation
    ↓
Performance Threshold Check

      Accuracy < 0.80 ?  ────────────
                                    |
      ↓ YES                         ↓ NO

Automatic Retraining        
      ↓                           
Train Updated Model          
      ↓                           
Log Results to MLflow        
      ↓                             
      └────────────→         Build Docker Image
                                    ↓
                            Push to DockerHub
                                    ↓
                            Update Kubernetes Manifest
                                    ↓
                            ArgoCD Detects Changes
                                    ↓
                            Sync Kind Cluster
                                    ↓
                           Flask API Updated


---

# Technology Stack

## Machine Learning

- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## MLOps

- GitHub Actions
- MLflow
- Docker
- DockerHub
- Kubernetes (Kind)
- ArgoCD

## API

- Flask

## Testing

- Pytest

---

# Repository Structure

sentiment-mlops-pipeline/

├── .github/
│   └── workflows/
│       ├── artifact-storage.yml
│       ├── dockerhub.yml
│       ├── mlops-pipeline.yml
│       ├── retrain.yml
│       └── tests.yml
│
├── app/
│   ├── app.py
│   └── __init__.py
|
├── data/
│   └── raw/
│       └── imdb_dataset.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── monitor_and_retrain.py
│
├── tests/
│   └── test_api.py
│
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metrics.json
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── argocd/
│   └── argocd-app.yaml
│
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── requirements.txt
└── README.md


---

# Pipeline Stages

## 1. Data Acquisition & Preprocessing

The preprocessing stage:

- Loads sentiment data
- Cleans text data
- Removes missing values
- Creates validation data
- Converts text into TF-IDF vectors

---

## 2. Model Training

The sentiment model is trained automatically using GitHub Actions.

Outputs:

- Trained model
- Accuracy score
- Performance metrics


---

## 3. MLflow Experiment Tracking

MLflow tracks:

- Accuracy
- Hyperparameters
- Training runs
- Model artifacts
- Retraining status


Run locally:

```bash
python -m mlflow ui --port 5001
```

Open:

```text
http://localhost:5001
```

---

# Continuous Integration (CI)

Triggered on:

- main
- develop
- feature/*

CI process:

1. Install dependencies
2. Run Pytest tests
3. Preprocess data
4. Train model

Unit tests include:

- Home endpoint validation
- Invalid route handling
- Prediction endpoint testing

Purpose:

- Detect failures before deployment
- Ensure code quality

---

# Continuous Deployment (CD)

After successful workflow execution:

1. Docker image builds automatically
2. Image pushed to DockerHub
3. Kubernetes deployment manifest updated
4. ArgoCD detects Git changes
5. Kind cluster updates automatically

---

# Continuous Training (CT)

Model monitoring evaluates performance using a validation dataset.

Current threshold:

```python
THRESHOLD = 0.80
```

Workflow:

Evaluate Model
      ↓
Accuracy below threshold?
      ↓
Yes
      ↓
Trigger Retraining
      ↓
Log metrics to MLflow
      ↓
Deploy updated model


Retraining also runs automatically every:

- 6 hours*
*adjustable in code



# Model Artefact Storage

A dedicated workflow stores trained model artefacts.

Stored artifacts:

- sentiment_model.pkl
- tfidf_vectorizer.pkl

Purpose:

- Model reproducibility
- Version tracking
- Easier deployment

---

# Kubernetes Deployment

The application is deployed using:

- Docker containerisation
- Kubernetes Deployment
- Kubernetes Service
- Kind Kubernetes cluster
- ArgoCD for GitOps synchronisation


Useful commands:

Deploy:

```bash
kubectl apply -f k8s/
```

Check resources:

```bash
kubectl get all
```

---

# GitOps with ArgoCD

ArgoCD continuously monitors GitHub.

When deployment manifests change:

- Detects updates
- Synchronizes cluster state
- Applies deployment automatically

Benefits:

- Git becomes the source of truth
- Automated deployment
- Easy rollback capability

---

# Branching Strategy

Branch structure:

Main branch:

- Production-ready code

Develop branch:

- Integration branch

Feature branches:

- Individual feature development


Workflow:

Feature Branch
      ↓
Develop Branch
      ↓
GitHub Actions Tests
      ↓
Main Branch
      ↓
Deployment


---

# Example API Request

Input:

```json
{
    "text":"I love this movie"
}
```

Output:

```json
{
    "prediction":"positive"
}
```

---

# Future Improvements

- Add Prometheus for real-time monitoring
- Add Grafana dashboards for visual metrics
- Implement model drift detection
- Deploy to a cloud platform (AWS, Azure or GCP)
- Add automatic rollback for failed deployments

---

# Author

Ross Moroney
L00196752

Big Data Architecture – MLOps Assessment Project CA3
Submission Date: 24/5/2026
