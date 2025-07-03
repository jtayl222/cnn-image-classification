import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
from torchvision import transforms
from predict import predictor
from unittest.mock import MagicMock

def test_predictor_file_not_found():
    # Should handle file not found gracefully
    model = MagicMock()
    top_p, top_class = predictor("nonexistent.jpg", model, topk=5, device="cpu")
    assert top_p is None
    assert top_class is None

def test_predictor_valid_image(tmp_path):
    from PIL import Image
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (256, 256), color="blue")
    img.save(img_path)
    # Mock model
    class DummyModel:
        def __init__(self):
            self.classes = ["a", "b", "c", "d", "e"]
        def forward(self, x):
            return torch.log(torch.ones((1, 5)))
        def to(self, device):
            return self
        def eval(self):
            return self
    model = DummyModel()
    top_p, top_class = predictor(str(img_path), model, topk=5, device="cpu")
    assert isinstance(top_p, list)
    assert isinstance(top_class, list)
    assert len(top_p) == 5
    assert len(top_class) == 5 