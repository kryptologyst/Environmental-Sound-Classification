#!/usr/bin/env python3
"""
Inference script for environmental sound classification.

This script loads a trained model and performs inference on audio files.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import librosa

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.crnn import CRNNModel
from src.features.melspectrogram import MelSpectrogramExtractor
from src.utils.device import get_device, set_seed


# ESC-50 class names
CLASS_NAMES = [
    "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
    "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
    "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
    "crickets", "crow", "crying_baby", "dog", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine", "fireworks", "footsteps",
    "frog", "glass_breaking", "hand_saw", "helicopter", "hen",
    "insects", "keyboard_typing", "laughing", "lawn_mower", "mouse_click",
    "panting", "pig", "pouring_water", "rain", "rooster",
    "sea_waves", "sheep", "siren", "snoring", "thunderstorm",
    "toilet_flush", "train", "vacuum_cleaner", "washing_machine", "water_drops"
]


def load_model(checkpoint_path: str, device: str = "auto") -> Tuple[CRNNModel, MelSpectrogramExtractor]:
    """Load the trained model and feature extractor."""
    device = get_device(device)
    
    # Create model
    model = CRNNModel(
        input_channels=1,
        num_classes=len(CLASS_NAMES),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )
    
    # Load checkpoint if available
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Create feature extractor
    feature_extractor = MelSpectrogramExtractor(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=11025,
        power=2.0,
        normalized=True,
    )
    
    return model, feature_extractor


def preprocess_audio(audio_path: str, sample_rate: int = 22050) -> torch.Tensor:
    """Load and preprocess audio file."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Truncate or pad to 5 seconds
    target_length = int(sample_rate * 5.0)
    if len(audio) > target_length:
        # Random crop
        start = np.random.randint(0, len(audio) - target_length + 1)
        audio = audio[start:start + target_length]
    else:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)))
    
    return torch.from_numpy(audio).float()


def predict_sound(
    model: CRNNModel, 
    feature_extractor: MelSpectrogramExtractor, 
    audio_tensor: torch.Tensor,
    top_k: int = 5
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Predict the environmental sound class."""
    device = next(model.parameters()).device
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(audio_tensor.unsqueeze(0).to(device))
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    predicted_category = CLASS_NAMES[predicted_class]
    all_probabilities = probabilities[0].cpu().numpy()
    
    # Get top-K predictions
    top_k_indices = np.argsort(all_probabilities)[-top_k:][::-1]
    top_k_results = [(CLASS_NAMES[i], all_probabilities[i]) for i in top_k_indices]
    
    return predicted_category, confidence, top_k_results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Environmental Sound Classification Inference")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--top-k", type=int, default=5, 
                       help="Number of top predictions to show")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    model, feature_extractor = load_model(args.checkpoint, args.device)
    
    # Load and preprocess audio
    print(f"Loading audio: {args.audio_file}")
    audio_tensor = preprocess_audio(args.audio_file)
    
    # Make prediction
    print("Classifying sound...")
    predicted_category, confidence, top_k_results = predict_sound(
        model, feature_extractor, audio_tensor, args.top_k
    )
    
    # Display results
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"File: {args.audio_file}")
    print(f"Predicted Sound: {predicted_category}")
    print(f"Confidence: {confidence:.3f}")
    print("\nTop-K Predictions:")
    for i, (category, prob) in enumerate(top_k_results, 1):
        print(f"  {i}. {category}: {prob:.3f}")
    print("="*50)


if __name__ == "__main__":
    main()
