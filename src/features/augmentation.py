"""Audio data augmentation modules."""

import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.
    
    Args:
        time_mask_param: Maximum length of time mask
        freq_mask_param: Maximum length of frequency mask
        num_time_masks: Number of time masks to apply
        num_freq_masks: Number of frequency masks to apply
        p: Probability of applying augmentation
    """
    
    def __init__(
        self,
        time_mask_param: int = 27,
        freq_mask_param: int = 8,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        p: float = 1.0,
    ):
        super().__init__()
        
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.p = p
        
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spec: Input spectrogram of shape (batch_size, n_mels, time_frames)
            
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.p:
            return spec
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec)
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec)
        
        return spec


class MixUp(nn.Module):
    """
    MixUp augmentation for audio classification.
    
    Args:
        alpha: MixUp parameter (beta distribution)
        p: Probability of applying MixUp
    """
    
    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p
    
    def forward(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            labels: Input labels tensor of shape (batch_size,)
            
        Returns:
            Tuple of (mixed_audio, mixed_labels, lambda_values)
        """
        if random.random() > self.p:
            return audio, labels, torch.ones(audio.size(0))
        
        batch_size = audio.size(0)
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        # Generate lambda values from beta distribution
        lambda_values = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size,))
        lambda_values = lambda_values.to(audio.device)
        
        # Mix audio
        mixed_audio = lambda_values.view(-1, 1) * audio + (1 - lambda_values.view(-1, 1)) * audio[indices]
        
        # Mix labels (one-hot encoding)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=labels.max().item() + 1).float()
        mixed_labels = lambda_values.view(-1, 1) * labels_one_hot + (1 - lambda_values.view(-1, 1)) * labels_one_hot[indices]
        
        return mixed_audio, mixed_labels, lambda_values


class TimeStretch(nn.Module):
    """
    Time stretching augmentation.
    
    Args:
        min_rate: Minimum stretch rate
        max_rate: Maximum stretch rate
        p: Probability of applying augmentation
    """
    
    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2, p: float = 0.5):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretching.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            
        Returns:
            Time-stretched audio tensor
        """
        if random.random() > self.p:
            return audio
        
        batch_size = audio.size(0)
        stretched_audio = []
        
        for i in range(batch_size):
            rate = random.uniform(self.min_rate, self.max_rate)
            audio_np = audio[i].cpu().numpy()
            
            # Use librosa for time stretching
            import librosa
            stretched = librosa.effects.time_stretch(audio_np, rate=rate)
            
            # Pad or truncate to original length
            if len(stretched) > len(audio_np):
                stretched = stretched[:len(audio_np)]
            else:
                stretched = np.pad(stretched, (0, len(audio_np) - len(stretched)))
            
            stretched_audio.append(torch.from_numpy(stretched))
        
        return torch.stack(stretched_audio).to(audio.device)


class PitchShift(nn.Module):
    """
    Pitch shifting augmentation.
    
    Args:
        min_semitones: Minimum pitch shift in semitones
        max_semitones: Maximum pitch shift in semitones
        p: Probability of applying augmentation
    """
    
    def __init__(self, min_semitones: float = -2.0, max_semitones: float = 2.0, p: float = 0.5):
        super().__init__()
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.p = p
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            
        Returns:
            Pitch-shifted audio tensor
        """
        if random.random() > self.p:
            return audio
        
        batch_size = audio.size(0)
        shifted_audio = []
        
        for i in range(batch_size):
            semitones = random.uniform(self.min_semitones, self.max_semitones)
            audio_np = audio[i].cpu().numpy()
            
            # Use librosa for pitch shifting
            import librosa
            shifted = librosa.effects.pitch_shift(audio_np, sr=22050, n_steps=semitones)
            
            shifted_audio.append(torch.from_numpy(shifted))
        
        return torch.stack(shifted_audio).to(audio.device)


class AddNoise(nn.Module):
    """
    Add Gaussian noise to audio.
    
    Args:
        min_snr: Minimum signal-to-noise ratio in dB
        max_snr: Maximum signal-to-noise ratio in dB
        p: Probability of applying augmentation
    """
    
    def __init__(self, min_snr: float = 10.0, max_snr: float = 30.0, p: float = 0.5):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add noise to audio.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples)
            
        Returns:
            Noisy audio tensor
        """
        if random.random() > self.p:
            return audio
        
        batch_size = audio.size(0)
        noisy_audio = []
        
        for i in range(batch_size):
            snr_db = random.uniform(self.min_snr, self.max_snr)
            signal_power = torch.mean(audio[i] ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            
            noise = torch.randn_like(audio[i]) * torch.sqrt(noise_power)
            noisy = audio[i] + noise
            
            noisy_audio.append(noisy)
        
        return torch.stack(noisy_audio).to(audio.device)


class ComposeAugmentations(nn.Module):
    """
    Compose multiple augmentations.
    
    Args:
        augmentations: List of augmentation modules
    """
    
    def __init__(self, augmentations: list):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)
    
    def forward(self, audio: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply composed augmentations.
        
        Args:
            audio: Input audio tensor
            labels: Input labels tensor (optional)
            
        Returns:
            Augmented audio and labels (if provided)
        """
        for aug in self.augmentations:
            if isinstance(aug, MixUp) and labels is not None:
                audio, labels, _ = aug(audio, labels)
            else:
                audio = aug(audio)
        
        if labels is not None:
            return audio, labels
        return audio
