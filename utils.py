import torch
from torch import nn, optim
from torchvision import transforms
import time


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
        # Optionally log metrics if mlflow is available in the calling script
        if 'mlflow' in globals():
            mlflow.log_metric("train_loss", epoch_loss, step=epoch+1)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch+1)
            mlflow.log_metric("accuracy", accuracy, step=epoch+1)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch+1)
            mlflow.log_metric("top_5_accuracy", top5_accuracy, step=epoch+1)
        scheduler.step()


def evaluate(model, dataloader, criterion, device):
    accuracy = 0
    test_loss = 0
    top5_accuracy = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            top_p, top_classes = ps.topk(5, dim=1)
            matches = top_classes.eq(labels.view(-1, 1))
            top5_correct = matches.any(dim=1).float().sum().item()
            top5_accuracy += top5_correct / labels.size(0)
    return test_loss / len(dataloader), accuracy / len(dataloader), top5_accuracy / len(dataloader)


def process_image(image, test_transforms):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Tensor '''
    from PIL import Image
    with Image.open(image) as im:
        im = test_transforms(im)
        return im 