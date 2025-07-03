import torch
from torch import nn, optim
from torchvision import datasets, transforms
import argparse
from load_model import load_model
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import time
import os
import tempfile
import sys



def print_env_var_hint():

    MLFLOW_ENV_VARS = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL",
        "MLFLOW_DB_USERNAME",
        "MLFLOW_DB_PASSWORD",
        "MLFLOW_FLASK_SECRET_KEY",
    ]
    descriptions = {
        "MLFLOW_TRACKING_URI": "MLflow tracking server URI (e.g., http://mlflow.local)",
        "MLFLOW_TRACKING_USERNAME": "Username for MLflow tracking server authentication",
        "MLFLOW_TRACKING_PASSWORD": "Password for MLflow tracking server authentication",
        "MLFLOW_S3_ENDPOINT_URL": "S3/MinIO endpoint for artifact storage (e.g., http://minio:9000)",
        "MLFLOW_DB_PASSWORD": "Password for MLflow backend database",
        "MLFLOW_FLASK_SECRET_KEY": "Flask secret key for MLflow UI session security",
        "MLFLOW_DB_USERNAME": "Username for MLflow backend database",
    }
    missing = [var for var in MLFLOW_ENV_VARS if var not in os.environ or not os.environ[var]]
    if missing:
        print("\n❌ Hint: You may be missing required environment variables:\n")
        for var in MLFLOW_ENV_VARS:
            status = "MISSING" if var in missing else "OK"
            desc = descriptions.get(var, "")
            print(f"  [{status:7}] {var:28} {desc}")
        print("\nPlease set all required environment variables before running this script.\n")
        print("They can be set in your shell or in a .env file loaded by your environment.")
        sys.exit(1)

def train(model, train_loader, validation_loader, criterion, optimizer, scheduler, epochs=5, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        validation_loss, accuracy, top5_accuracy = evaluate(model, validation_loader, criterion, device)
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {epoch_loss:.3f}.. "
              f"Validation loss: {validation_loss:.3f}.. "
              f"Test accuracy: {accuracy:.3f}.. "
              f"Epoch time: {epoch_time:.3f}s")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch+1)
        mlflow.log_metric("validation_loss", validation_loss, step=epoch+1)
        mlflow.log_metric("accuracy", accuracy, step=epoch+1)
        mlflow.log_metric("epoch_time", epoch_time, step=epoch+1)
        mlflow.log_metric("top_5_accuracy", top5_accuracy, step=epoch+1)
        
        scheduler.step()
        current_scheduler_step = scheduler.last_epoch
        mlflow.log_metric("scheduler_step", current_scheduler_step, step=epoch+1)


def evaluate(model, dataloader, criterion, device):
    accuracy = 0
    test_loss = 0
    top5_accuracy = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ## Calculating the accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            top_p, top_classes = ps.topk(5, dim=1)
            matches = top_classes.eq(labels.view(-1, 1))
            top5_correct = matches.any(dim=1).float().sum().item()
            top5_accuracy += top5_correct / labels.size(0)

    return test_loss / len(dataloader), accuracy / len(dataloader), top5_accuracy / len(dataloader)

def save_checkpoint(filepath, model):
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'classes': model.classes,
                  'input_size': model.classifier.fc1.in_features,
                  'fc2_size': model.classifier.fc2.in_features,
                  'output_size': model.classifier.fc2.out_features,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)


def init_data_loading(data_dir, batch_size=64):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
    print("training size =", len(train_data.samples), "validation size =", len(validation_data.samples))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    return train_loader, validation_loader

# Create a signature and input example for the model for MLflow logging
def create_signature(model, validation_loader, device):
    images, labels = next(iter(validation_loader))   # get a single batch from validation
    images = images.to(device)                 # ensure it's on CPU for logging

    model.eval()
    with torch.no_grad():
        outputs = model(images)               # forward pass on this batch

    # log one sample, you can do something like:
    single_input_example = images[0].unsqueeze(0)
    single_output = outputs[0].unsqueeze(0)

    # Convert to NumPy for signature inference
    signature = infer_signature(
        single_input_example.cpu().detach().numpy(),
        single_output.cpu().detach().numpy()
    )
    return signature, single_input_example
    
def main(
        data_dir="args.data_directory",
        save_dir="checkpoints",
        arch="vgg16",
        epochs=5,
        device="cpu",
        batch_size=64):
    
    # Example hyperparameter grid
    lr_values = [0.0001, 0.001, 0.01]
    hidden_units_values = [256, 512, 1024]

    if "MLFLOW_TRACKING_URI" not in os.environ:
        print("MLFLOW_TRACKING_URI environment variable is not set.")
        print_env_var_hint()
        exit(1)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        mlflow.set_experiment("cnn-flower-classifier")
    except mlflow.exceptions.MlflowException as e:
        print("ERROR setting MLflow experiment:", e)
        print_env_var_hint()
        exit(1)

    # Load data once, outside the hyperparameter loops
    train_loader, validation_loader = init_data_loading(data_dir, batch_size=batch_size)

    for learning_rate in lr_values:
        for hidden_units in hidden_units_values:
            # Load model with current hyperparameters
            model = load_model(arch, save_dir, hidden_units, train_loader=train_loader)
            
            with tempfile.TemporaryDirectory() as local_artifact_dir:
                print("local_artifact_dir =", local_artifact_dir)

                # Model checkpoint saved locally
                model_path = os.path.join(local_artifact_dir, "model")
                torch.save(model.state_dict(), model_path)

                with mlflow.start_run(run_name=f"lr={learning_rate}, hu={hidden_units}"):
                    mlflow.log_param("data_dir", data_dir)
                    mlflow.log_param("save_dir", save_dir)
                    mlflow.log_param("arch", arch)
                    mlflow.log_param("learning_rate", learning_rate)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("hidden_units", hidden_units)
                    mlflow.log_param("device", device)

                    criterion = nn.NLLLoss()
                    mlflow.log_param("criterion", criterion.__class__.__name__)

                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    mlflow.log_param("optimizer", optimizer.__class__.__name__)

                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                    mlflow.log_param("scheduler", scheduler.__class__.__name__)
                    mlflow.log_param("step_size", 5)
                    mlflow.log_param("gamma", 0.1)

                    train(model, train_loader, validation_loader, criterion, optimizer, scheduler, epochs=epochs, device=device)

                    signature, single_input_example = create_signature(model, validation_loader, device)
                    single_input_example_np = single_input_example.cpu().detach().numpy()
                    mlflow.pytorch.log_model(
                        model, 
                        artifact_path="model",
                        input_example=single_input_example_np,
                        signature=signature
                    )

                    if save_dir is not None:
                        save_checkpoint(save_dir, model)

if __name__ == '__main__':
    # check_env_vars()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a new network on a data set")
    parser.add_argument("data_directory", help="Directory containing train/{class}/example", default="flowers")
    parser.add_argument("--save_dir", help="checkpoint file", default=None)
    parser.add_argument("--arch", help="Example: vgg13", default="vgg16")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--cpu", action="store_true", help="Use CPU for training")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    args = parser.parse_args()

    # Device selection logic
    if args.cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested.")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA device available, using: cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"MPS device available, using: {device}")
    else:
        device = torch.device("cpu")
        print("No GPU/MPS device found. Using CPU.")

    print("Calling main()")
    main(
        data_dir=args.data_directory,
        save_dir=args.save_dir,
        arch=args.arch,
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size
    )
