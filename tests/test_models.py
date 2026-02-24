"""Unit tests for environmental sound classification."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.crnn import CRNNModel
from src.features.melspectrogram import MelSpectrogramExtractor
from src.features.augmentation import SpecAugment, MixUp
from src.losses.focal_loss import FocalLoss
from src.metrics.classification import ClassificationMetrics
from src.utils.device import get_device, set_seed
from src.utils.audio import load_audio, resample_audio


class TestCRNNModel:
    """Test CRNN model functionality."""
    
    def test_model_creation(self):
        """Test model creation with default parameters."""
        model = CRNNModel()
        assert isinstance(model, CRNNModel)
        assert model.num_classes == 50
        assert model.hidden_size == 128
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = CRNNModel(num_classes=10)
        batch_size = 4
        n_mels = 128
        time_frames = 100
        
        # Create dummy input (batch_size, channels, height, width)
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_embeddings(self):
        """Test model embedding extraction."""
        model = CRNNModel(num_classes=10)
        batch_size = 2
        n_mels = 128
        time_frames = 50
        
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        embeddings = model.get_embeddings(x)
        
        # Embeddings should have shape (batch_size, rnn_output_size)
        expected_size = model.hidden_size * 2 if model.bidirectional else model.hidden_size
        assert embeddings.shape == (batch_size, expected_size)


class TestMelSpectrogramExtractor:
    """Test mel-spectrogram feature extractor."""
    
    def test_extractor_creation(self):
        """Test extractor creation."""
        extractor = MelSpectrogramExtractor()
        assert isinstance(extractor, MelSpectrogramExtractor)
        assert extractor.sample_rate == 22050
        assert extractor.n_mels == 128
    
    def test_extractor_forward(self):
        """Test extractor forward pass."""
        extractor = MelSpectrogramExtractor()
        batch_size = 2
        audio_length = 22050  # 1 second at 22.05 kHz
        
        audio = torch.randn(batch_size, audio_length)
        features = extractor(audio)
        
        # Features should have shape (batch_size, n_mels, time_frames)
        assert features.shape[0] == batch_size
        assert features.shape[1] == extractor.n_mels
        assert features.shape[2] > 0  # Time frames
    
    def test_feature_shape_calculation(self):
        """Test feature shape calculation."""
        extractor = MelSpectrogramExtractor()
        audio_length = 22050
        
        n_mels, time_frames = extractor.get_feature_shape(audio_length)
        assert n_mels == extractor.n_mels
        assert time_frames > 0


class TestAugmentation:
    """Test data augmentation modules."""
    
    def test_spec_augment(self):
        """Test SpecAugment augmentation."""
        spec_augment = SpecAugment()
        batch_size = 2
        n_mels = 128
        time_frames = 100
        
        spec = torch.randn(batch_size, n_mels, time_frames)
        augmented = spec_augment(spec)
        
        assert augmented.shape == spec.shape
        assert not torch.isnan(augmented).any()
    
    def test_mixup(self):
        """Test MixUp augmentation."""
        mixup = MixUp(alpha=0.2)
        batch_size = 4
        audio_length = 1000
        num_classes = 10
        
        audio = torch.randn(batch_size, audio_length)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        mixed_audio, mixed_labels, lam = mixup(audio, labels)
        
        assert mixed_audio.shape == audio.shape
        assert mixed_labels.shape == (batch_size, num_classes)
        assert lam.shape == (batch_size,)


class TestLossFunctions:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test Focal loss."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        batch_size = 4
        num_classes = 10
        
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = focal_loss(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_focal_loss_one_hot(self):
        """Test Focal loss with one-hot targets."""
        focal_loss = FocalLoss()
        batch_size = 2
        num_classes = 5
        
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).float()
        
        loss = focal_loss(logits, targets_one_hot)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        num_classes = 5
        class_names = [f"class_{i}" for i in range(num_classes)]
        metrics_calc = ClassificationMetrics(num_classes, class_names)
        
        batch_size = 20
        predictions = np.random.randint(0, num_classes, batch_size)
        targets = np.random.randint(0, num_classes, batch_size)
        probabilities = np.random.rand(batch_size, num_classes)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        metrics = metrics_calc.compute_metrics(predictions, targets, probabilities)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_micro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        metrics_calc = ClassificationMetrics(3, ["A", "B", "C"])
        
        predictions = np.array([0, 1, 2, 0, 1])
        targets = np.array([0, 1, 2, 1, 0])
        
        cm = metrics_calc.compute_confusion_matrix(predictions, targets)
        
        assert cm.shape == (3, 3)
        assert cm.sum() == len(predictions)


class TestUtils:
    """Test utility functions."""
    
    def test_device_selection(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device_cpu = get_device("cpu")
        assert device_cpu.type == "cpu"
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be the same
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_resample_audio(self):
        """Test audio resampling."""
        # Create dummy audio
        orig_sr = 44100
        target_sr = 22050
        duration = 1.0  # seconds
        
        audio = np.random.randn(int(orig_sr * duration))
        resampled = resample_audio(audio, orig_sr, target_sr)
        
        assert len(resampled) == int(target_sr * duration)
        assert not np.isnan(resampled).any()


class TestIntegration:
    """Integration tests."""
    
    def test_model_with_extractor(self):
        """Test model with feature extractor."""
        extractor = MelSpectrogramExtractor()
        model = CRNNModel(num_classes=10)
        
        batch_size = 2
        audio_length = 22050
        
        # Create dummy audio
        audio = torch.randn(batch_size, audio_length)
        
        # Extract features
        features = extractor(audio)
        
        # Model forward pass
        logits = model(features)
        
        assert logits.shape == (batch_size, 10)
        assert not torch.isnan(logits).any()
    
    def test_training_step_simulation(self):
        """Simulate a training step."""
        model = CRNNModel(num_classes=5)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        batch_size = 4
        n_mels = 128
        time_frames = 50
        
        # Dummy data
        features = torch.randn(batch_size, 1, n_mels, time_frames)
        labels = torch.randint(0, 5, (batch_size,))
        
        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])
