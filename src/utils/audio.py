"""Audio processing utilities."""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with librosa.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono
        normalize: Normalize audio
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(
            str(file_path),
            sr=sample_rate,
            mono=mono,
            res_type="kaiser_fast"
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        warnings.warn(f"Error loading {file_path}: {e}")
        # Return silence if loading fails
        duration = 5.0  # Default duration
        audio = np.zeros(int(sample_rate * duration))
        return audio, sample_rate


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = 22050,
    format: str = "WAV"
) -> None:
    """
    Save audio array to file.
    
    Args:
        audio: Audio array
        file_path: Output file path
        sample_rate: Sample rate
        format: Audio format
    """
    sf.write(str(file_path), audio, sample_rate, format=format)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(
    audio: np.ndarray,
    top_db: float = 20.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Trim silence from audio.
    
    Args:
        audio: Input audio array
        top_db: Silence threshold in dB
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Trimmed audio array
    """
    return librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]


def get_audio_info(file_path: Union[str, Path]) -> dict:
    """
    Get audio file information.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    try:
        info = sf.info(str(file_path))
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype
        }
    except Exception as e:
        warnings.warn(f"Error getting info for {file_path}: {e}")
        return {
            "duration": 0.0,
            "sample_rate": 22050,
            "channels": 1,
            "frames": 0,
            "format": "UNKNOWN",
            "subtype": "UNKNOWN"
        }


def find_audio_files(
    directory: Union[str, Path],
    extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
) -> List[Path]:
    """
    Find all audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of audio file extensions
        
    Returns:
        List of audio file paths
    """
    directory = Path(directory)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory.rglob(f"*{ext}"))
        audio_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    return sorted(audio_files)
