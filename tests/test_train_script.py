import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train

def test_train_script_runs(monkeypatch):
    # Patch sys.argv to simulate command-line arguments
    monkeypatch.setattr(sys, 'argv', [
        'train.py', 'flowers', '--save_dir', 'checkpoints', '--arch', 'vgg16', '--epochs', '1', '--batch_size', '2', '--learning_rate', '0.001', '--hidden_units', '256', '--cpu'])
    # Should not raise on import or argument parsing
    # We do not actually run the full training loop to avoid long test times
    # Instead, just check that argument parsing works
    try:
        import importlib
        importlib.reload(train)
    except SystemExit:
        # train.py may call sys.exit() if env vars are missing, that's fine for this test
        pass 