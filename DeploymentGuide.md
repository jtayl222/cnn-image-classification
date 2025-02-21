# Deployment Guide

This document describes how to use the existing `predict.py` and `train.py` scripts in a real-life example, including their integration into a CI/CD pipeline and deployment to a production microservice. Additionally, it discusses the role of MLFlow in the CI/CD pipeline.

## Using `train.py` and `predict.py`

### `train.py`

The `train.py` script is responsible for training the image classifier model. It includes steps for loading and preprocessing data, defining the model architecture, training the model, and saving the trained model checkpoint.

### `predict.py`

The `predict.py` script handles the prediction of image classes using the trained model. It includes steps for loading the model checkpoint, preprocessing the input image, and predicting the class of the input image.

## CI/CD Pipeline

### Continuous Integration (CI)

1. **Code Quality Checks**: Use tools like `flake8` for linting and `pytest` for running unit tests to ensure code quality and correctness.
2. **Automated Testing**: Set up a CI service like GitHub Actions, Travis CI, or CircleCI to automatically run tests on the `train.py` and `predict.py` scripts whenever changes are pushed to the repository.
3. **Model Training**: Automate the training process by running `train.py` in the CI pipeline. Save the trained model checkpoint as an artifact for later use.

### Continuous Deployment (CD)

1. **Model Versioning**: Use MLFlow to track and version the trained models. MLFlow can log parameters, metrics, and artifacts (model checkpoints).
2. **Model Registry**: Register the trained model in the MLFlow Model Registry. This allows for easy management and deployment of different model versions.
3. **Deployment**: Deploy the model to a production environment using a microservice framework like Flask or FastAPI. The microservice will load the model checkpoint and serve predictions via an API.

## Role of MLFlow in the CI/CD Pipeline

1. **Experiment Tracking**: Use MLFlow to log experiments during the training process. This includes logging hyperparameters, metrics, and model artifacts.
2. **Model Registry**: Register the best-performing model in the MLFlow Model Registry. This ensures that the model is versioned and can be easily deployed.
3. **Deployment**: Use MLFlow's deployment capabilities to deploy the model to a production environment. MLFlow supports various deployment targets, including Docker, Kubernetes, and cloud services.

## Real-Life Example

### Step 1: Setting Up the CI/CD Pipeline

1. **Create a GitHub Repository**: Initialize a GitHub repository and add the `train.py` and `predict.py` scripts.
2. **Set Up GitHub Actions**: Create a `.github/workflows/ci.yml` file to define the CI workflow. This workflow will include steps for installing dependencies, running tests, and training the model.

```yaml
name: CI Pipeline

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest

    - name: Train model
      run: |
        python train.py
```

### Step 2: Model Versioning with MLFlow

import mlflow
import mlflow.pytorch

# Inside the training function
with mlflow.start_run():
    # Log parameters, metrics, and model
    mlflow.log_param("learning_rate", 0.003)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.pytorch.log_model(model, "model")
```

2. **Register Model**: After training, register the model in the MLFlow Model Registry.

```python
# Register the model
model_uri = "runs:/{}/model".format(run_id)
mlflow.register_model(model_uri, "ImageClassifierModel")
```

### Step 3: Deploying the Model

1. **Create a Flask App**: Create a `app.py` file to serve the model predictions using Flask.

```python
from flask import Flask, request, jsonify
import mlflow.pytorch

app = Flask(__name__)

# Load the model
model = mlflow.pytorch.load_model("models:/ImageClassifierModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data['image_path']
    top_p, top_class = predict(image_path, model)
    return jsonify({'probabilities': top_p, 'classes': top_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

2. **Dockerize the Application**: Create a `Dockerfile` to containerize the Flask application.

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

3. **Deploy to Kubernetes**: Create Kubernetes deployment and service files to deploy the Docker container.

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-classifier
spec:
  replicas: 2
  selector:
    matchLabels:
      app: image-classifier
  template:
    metadata:
      labels:
        app: image-classifier
    spec:
      containers:
      - name: image-classifier
        image: your-docker-image
        ports:
        - containerPort: 5000
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: image-classifier
spec:
  selector:
    app: image-classifier
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

By following these steps, you can effectively use `train.py` and `predict.py` in a real-life example, integrating them into a CI/CD pipeline and deploying the model to a production microservice.

To implement logging and monitoring for the deployed model, you can use a combination of tools and techniques. Here is a step-by-step guide:

### 1. Logging with Python's `logging` Module

First, set up logging in your `app.py` file to capture important events and errors.

```python
import logging
from flask import Flask, request, jsonify
import mlflow.pytorch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
model = mlflow.pytorch.load_model("models:/ImageClassifierModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data['image_path']
    logger.info(f"Received prediction request for image: {image_path}")
    try:
        top_p, top_class = predict(image_path, model)
        logger.info(f"Prediction successful for image: {image_path}")
        return jsonify({'probabilities': top_p, 'classes': top_class})
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. Monitoring with Prometheus and Grafana

#### Step 1: Install Prometheus Client

Install the Prometheus client library for Python.

```sh
pip install prometheus_client
```

#### Step 2: Integrate Prometheus with Flask

Modify your `app.py` to expose metrics.

```python
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Define Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
PREDICTION_COUNT = Counter('prediction_count', 'Total number of predictions')
ERROR_COUNT = Counter('error_count', 'Total number of errors')

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.inc()
    data = request.json
    image_path = data['image_path']
    logger.info(f"Received prediction request for image: {image_path}")
    try:
        top_p, top_class = predict(image_path, model)
        PREDICTION_COUNT.inc()
        logger.info(f"Prediction successful for image: {image_path}")
        return jsonify({'probabilities': top_p, 'classes': top_class})
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
```

#### Step 3: Set Up Prometheus

Create a `prometheus.yml` configuration file.

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'flask_app'
    static_configs:
      - targets: ['localhost:5000']
```

Run Prometheus with the configuration file.

```sh
prometheus --config.file=prometheus.yml
```

#### Step 4: Set Up Grafana

1. Install and run Grafana.
2. Add Prometheus as a data source in Grafana.
3. Create dashboards to visualize the metrics collected by Prometheus.

By following these steps, you can implement logging and monitoring for your deployed model, ensuring you can track its performance and detect any issues in real-time.