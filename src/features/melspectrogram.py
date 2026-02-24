"""Audio feature extraction modules."""

import warnings
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    """
    Mel-spectrogram feature extractor.
    
    Args:
        sample_rate: Sample rate of input audio
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Exponent for the magnitude spectrogram
        normalized: Whether to normalize the mel-spectrogram
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = True,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalized = normalized
        
        # Create mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            power=power,
            normalized=normalized,
        )
        
        # Convert to log scale
        self.log_transform = T.AmplitudeToDB()
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram features.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            
        Returns:
            Mel-spectrogram tensor of shape (batch_size, n_mels, time_frames)
        """
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Extract mel-spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel_spec = self.log_transform(mel_spec)
        
        return log_mel_spec
    
    def get_feature_shape(self, audio_length: int) -> Tuple[int, int]:
        """
        Get the shape of features for given audio length.
        
        Args:
            audio_length: Length of input audio in samples
            
        Returns:
            Tuple of (n_mels, time_frames)
        """
        time_frames = (audio_length - self.n_fft) // self.hop_length + 1
        return (self.n_mels, time_frames)


class MFCCExtractor(nn.Module):
    """
    MFCC feature extractor.
    
    Args:
        sample_rate: Sample rate of input audio
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
        # Create MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "f_min": fmin,
                "f_max": self.fmax,
            }
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            
        Returns:
            MFCC tensor of shape (batch_size, n_mfcc, time_frames)
        """
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Extract MFCC
        mfcc = self.mfcc_transform(audio)
        
        return mfcc


class SpectralFeaturesExtractor(nn.Module):
    """
    Extract various spectral features.
    
    Args:
        sample_rate: Sample rate of input audio
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Create spectral centroid transform
        self.spectral_centroid = T.SpectralCentroid(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        # Create spectral rolloff transform
        self.spectral_rolloff = T.SpectralRolloff(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        # Create zero crossing rate transform
        self.zcr = T.ZeroCrossingRate(
            frame_length=n_fft,
            hop_length=hop_length,
        )
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract spectral features.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            
        Returns:
            Dictionary of spectral features
        """
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        features = {
            "spectral_centroid": self.spectral_centroid(audio),
            "spectral_rolloff": self.spectral_rolloff(audio),
            "zero_crossing_rate": self.zcr(audio),
        }
        
        return features


def extract_features_numpy(
    audio: np.ndarray,
    sample_rate: int = 22050,
    feature_type: str = "melspectrogram",
    **kwargs
) -> np.ndarray:
    """
    Extract features using librosa (numpy backend).
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        feature_type: Type of features to extract
        **kwargs: Additional arguments for feature extraction
        
    Returns:
        Feature array
    """
    if feature_type == "melspectrogram":
        return librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            **kwargs
        )
    elif feature_type == "mfcc":
        return librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            **kwargs
        )
    elif feature_type == "spectral_centroid":
        return librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate,
            **kwargs
        )
    elif feature_type == "spectral_rolloff":
        return librosa.feature.spectral_rolloff(
            y=audio,
            sr=sample_rate,
            **kwargs
        )
    elif feature_type == "zero_crossing_rate":
        return librosa.feature.zero_crossing_rate(
            y=audio,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
