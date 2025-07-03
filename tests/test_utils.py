import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
from torch import nn, optim
from torchvision import transforms
from utils import train, evaluate, process_image
from unittest.mock import MagicMock
import tempfile
import os

class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Match input shape: 3x224x224 -> 150528
        self.fc = nn.Linear(3*224*224, num_classes)
        self.classes = list(range(num_classes))
        self.class_to_idx = {str(i): i for i in range(num_classes)}
    def forward(self, x):
        return torch.log_softmax(self.fc(x.view(x.size(0), -1)), dim=1)
    def to(self, device):
        # Simulate .to() for compatibility
        return self
    def eval(self):
        return self

@pytest.fixture
def dummy_data():
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 2, (8,))
    dataset = list(zip(x, y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    return loader

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def dummy_criterion():
    return nn.NLLLoss()

@pytest.fixture
def dummy_optimizer(dummy_model):
    return optim.Adam(dummy_model.parameters(), lr=0.001)

@pytest.fixture
def dummy_scheduler(dummy_optimizer):
    return optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1)

def test_train_runs(dummy_model, dummy_data, dummy_criterion, dummy_optimizer, dummy_scheduler):
    # Should run without error
    train(dummy_model, dummy_data, dummy_data, dummy_criterion, dummy_optimizer, dummy_scheduler, epochs=1, device="cpu")

def test_evaluate_returns_metrics(dummy_model, dummy_data, dummy_criterion):
    loss, acc, top5 = evaluate(dummy_model, dummy_data, dummy_criterion, device="cpu")
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert isinstance(top5, float)

def test_process_image(tmp_path):
    from PIL import Image
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (256, 256), color="red")
    img.save(img_path)
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = process_image(str(img_path), test_transforms)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == 224 and tensor.shape[2] == 224

def test_process_image_file_not_found():
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        result = process_image("nonexistent.jpg", test_transforms)
        assert result is None
    except FileNotFoundError:
        # If process_image does not catch, this will catch for test pass
        pass 