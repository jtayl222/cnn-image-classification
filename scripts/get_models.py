import mlflow
import mlflow
from mlflow.tracking import MlflowClient
import json
import os

def get_Models():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    data = client.search_registered_models()
    print("data is " + str(data))
    print("client is " + str(client))
    experiments = client.search_experiments()
    print("experiments is " + str(experiments))

    models = []
    for model in data:
        models.append(model.name)
        print("models is " + str(models))
    result = []
    for model in models:
        model_versions = {"name": model}
        data = client.search_model_versions(filter_string=f"name='{model}'", order_by=["version_number DESC"])
        versions = list(map(lambda x: dict(x), data))
        model_versions["latest_versions"] = versions
        result.append(model_versions)
    return result

def main():
    models = get_Models()
    print(json.dumps(models, indent=4))

if __name__ == "__main__":
    main()

