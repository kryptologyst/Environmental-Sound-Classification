"""
Streamlit demo for Environmental Sound Classification.

This demo allows users to upload audio files or record audio and classify
environmental sounds using the trained model.
"""

import io
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

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

# Model configuration
MODEL_CONFIG = {
    "input_channels": 1,
    "num_classes": 50,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
}

# Feature extractor configuration
FEATURE_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 11025,
    "power": 2.0,
    "normalized": True,
}


@st.cache_resource
def load_model(checkpoint_path: str) -> Tuple[CRNNModel, MelSpectrogramExtractor]:
    """Load the trained model and feature extractor."""
    device = get_device("cpu")  # Use CPU for demo
    
    # Create model
    model = CRNNModel(**MODEL_CONFIG)
    
    # Load checkpoint if available
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Create feature extractor
    feature_extractor = MelSpectrogramExtractor(**FEATURE_CONFIG)
    
    return model, feature_extractor


def preprocess_audio(audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
    """Preprocess audio for model input."""
    # Resample if necessary
    if sample_rate != FEATURE_CONFIG["sample_rate"]:
        audio_data = librosa.resample(
            audio_data, 
            orig_sr=sample_rate, 
            target_sr=FEATURE_CONFIG["sample_rate"]
        )
    
    # Truncate or pad to 5 seconds
    target_length = int(FEATURE_CONFIG["sample_rate"] * 5.0)
    if len(audio_data) > target_length:
        # Random crop
        start = np.random.randint(0, len(audio_data) - target_length + 1)
        audio_data = audio_data[start:start + target_length]
    else:
        # Pad with zeros
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
    
    return torch.from_numpy(audio_data).float()


def predict_sound(model: CRNNModel, feature_extractor: MelSpectrogramExtractor, 
                  audio_tensor: torch.Tensor) -> Tuple[str, float, np.ndarray]:
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
    
    return predicted_category, confidence, all_probabilities


def plot_spectrogram(audio_data: np.ndarray, sample_rate: int) -> None:
    """Plot mel-spectrogram."""
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=FEATURE_CONFIG["n_fft"],
        hop_length=FEATURE_CONFIG["hop_length"],
        n_mels=FEATURE_CONFIG["n_mels"],
        fmin=FEATURE_CONFIG["fmin"],
        fmax=FEATURE_CONFIG["fmax"],
        power=FEATURE_CONFIG["power"]
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_mel_spec,
        sr=sample_rate,
        hop_length=FEATURE_CONFIG["hop_length"],
        x_axis="time",
        y_axis="mel",
        fmin=FEATURE_CONFIG["fmin"],
        fmax=FEATURE_CONFIG["fmax"]
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-Spectrogram")
    plt.tight_layout()
    
    return plt.gcf()


def plot_predictions(probabilities: np.ndarray, top_k: int = 10) -> None:
    """Plot top-K predictions."""
    # Get top-K predictions
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    top_k_names = [CLASS_NAMES[i] for i in top_k_indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(top_k_names)), top_k_probs)
    plt.yticks(range(len(top_k_names)), top_k_names)
    plt.xlabel("Probability")
    plt.title(f"Top-{top_k} Predictions")
    plt.xlim(0, 1)
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, top_k_probs)):
        plt.text(prob + 0.01, i, f"{prob:.3f}", va="center")
    
    plt.tight_layout()
    
    return plt.gcf()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Environmental Sound Classification",
        page_icon="🔊",
        layout="wide"
    )
    
    # Title and description
    st.title("🔊 Environmental Sound Classification")
    st.markdown("""
    This demo classifies environmental sounds using a deep learning model trained on the ESC-50 dataset.
    Upload an audio file or record audio to get started!
    """)
    
    # Privacy disclaimer
    st.warning("""
    **Privacy Notice**: This is a research demonstration. Audio files are processed locally and not stored.
    This system is not intended for biometric identification or production use.
    """)
    
    # Load model
    checkpoint_path = "checkpoints/best_model.pt"
    model, feature_extractor = load_model(checkpoint_path)
    
    # Sidebar for input options
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Audio File", "Record Audio"]
    )
    
    audio_data = None
    sample_rate = None
    filename = None
    
    if input_method == "Upload Audio File":
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                filename = uploaded_file.name
                st.sidebar.success(f"Loaded: {filename}")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
            finally:
                # Clean up temp file
                Path(tmp_path).unlink()
    
    elif input_method == "Record Audio":
        # Audio recording
        st.sidebar.info("Click the microphone to start recording")
        
        # Note: Streamlit doesn't have built-in audio recording
        # This would require additional JavaScript or external libraries
        st.sidebar.warning("Audio recording feature requires additional setup.")
    
    # Main content
    if audio_data is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Audio Analysis")
            
            # Audio player
            st.audio(audio_data, sample_rate=sample_rate)
            
            # Audio info
            duration = len(audio_data) / sample_rate
            st.info(f"""
            **Audio Information:**
            - Duration: {duration:.2f} seconds
            - Sample Rate: {sample_rate} Hz
            - Channels: Mono
            """)
            
            # Spectrogram
            st.subheader("Mel-Spectrogram")
            fig_spec = plot_spectrogram(audio_data, sample_rate)
            st.pyplot(fig_spec)
        
        with col2:
            st.subheader("Classification Results")
            
            # Preprocess audio
            audio_tensor = preprocess_audio(audio_data, sample_rate)
            
            # Make prediction
            with st.spinner("Classifying sound..."):
                predicted_category, confidence, probabilities = predict_sound(
                    model, feature_extractor, audio_tensor
                )
            
            # Display results
            st.success(f"**Predicted Sound:** {predicted_category}")
            st.info(f"**Confidence:** {confidence:.3f}")
            
            # Top predictions
            st.subheader("Top Predictions")
            fig_pred = plot_predictions(probabilities, top_k=10)
            st.pyplot(fig_pred)
            
            # Detailed results
            with st.expander("Detailed Results"):
                results_df = []
                for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                    results_df.append({
                        "Rank": i + 1,
                        "Class": name,
                        "Probability": prob
                    })
                
                results_df = sorted(results_df, key=lambda x: x["Probability"], reverse=True)
                st.dataframe(results_df, use_container_width=True)
    
    else:
        # Instructions
        st.info("""
        **How to use this demo:**
        
        1. **Upload an audio file** using the sidebar (WAV, MP3, FLAC, M4A, OGG)
        2. The system will automatically:
           - Load and preprocess the audio
           - Extract mel-spectrogram features
           - Classify the environmental sound
           - Display results with confidence scores
        
        3. **Supported sounds** include:
           - Animals (cat, dog, cow, sheep, etc.)
           - Nature (rain, thunderstorm, sea waves, etc.)
           - Human sounds (laughing, coughing, clapping, etc.)
           - Machines (car horn, engine, vacuum cleaner, etc.)
           - And many more!
        
        **Note**: The model works best with 5-second audio clips. Longer files will be cropped.
        """)
        
        # Example sounds
        st.subheader("Example Environmental Sounds")
        example_sounds = [
            "Airplane", "Car Horn", "Dog Barking", "Rain", "Thunderstorm",
            "Sea Waves", "Birds Chirping", "Clock Tick", "Footsteps", "Laughing"
        ]
        
        cols = st.columns(3)
        for i, sound in enumerate(example_sounds):
            with cols[i % 3]:
                st.write(f"• {sound}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this project:**
    
    This environmental sound classification system uses a CRNN (Convolutional Recurrent Neural Network) 
    architecture trained on the ESC-50 dataset. The model combines CNN layers for feature extraction 
    with RNN layers for temporal modeling.
    
    **Technical Details:**
    - Model: CRNN with 2-layer bidirectional LSTM
    - Features: 128-bin mel-spectrogram
    - Input: 5-second audio clips at 22.05 kHz
    - Classes: 50 environmental sound categories
    
    **Disclaimer:** This is a research demonstration. The model may not perform optimally on all audio 
    types and should not be used for critical applications without proper validation.
    """)


if __name__ == "__main__":
    main()
