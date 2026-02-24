"""CRNN (CNN-RNN) model for audio classification."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) model for audio classification.
    
    This model combines CNN layers for feature extraction with RNN layers for
    temporal modeling, followed by a classification head.
    
    Args:
        input_channels: Number of input channels (default: 1 for mono audio)
        num_classes: Number of output classes
        hidden_size: Hidden size of RNN layers
        num_layers: Number of RNN layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional RNN
        feature_dim: Dimension of CNN features (auto-computed if None)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # Global average pooling in frequency dimension
        )
        
        # Calculate feature dimension after CNN
        if feature_dim is None:
            # This will be set during forward pass
            self.feature_dim = 256
        else:
            self.feature_dim = feature_dim
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Calculate RNN output size
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               where height is n_mels and width is time_frames
               
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)  # (batch_size, 256, 1, time_frames)
        
        # Reshape for RNN: (batch_size, time_frames, feature_dim)
        conv_features = conv_features.squeeze(2).transpose(1, 2)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)  # (batch_size, time_frames, rnn_output_size)
        
        # Global average pooling over time dimension
        pooled_output = torch.mean(rnn_output, dim=1)  # (batch_size, rnn_output_size)
        
        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings from the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               
        Returns:
            Feature embeddings of shape (batch_size, rnn_output_size)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)
        
        # Reshape for RNN
        conv_features = conv_features.squeeze(2).transpose(1, 2)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)
        
        # Global average pooling
        embeddings = torch.mean(rnn_output, dim=1)
        
        return embeddings
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               
        Returns:
            Attention weights of shape (batch_size, time_frames)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)
        
        # Reshape for RNN
        conv_features = conv_features.squeeze(2).transpose(1, 2)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)
        
        # Compute attention weights (simple average pooling weights)
        attention_weights = torch.softmax(torch.mean(rnn_output, dim=2), dim=1)
        
        return attention_weights


class CRNNWithAttention(nn.Module):
    """
    CRNN model with attention mechanism.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        hidden_size: Hidden size of RNN layers
        num_layers: Number of RNN layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional RNN
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # CNN feature extractor (same as CRNNModel)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Attention mechanism
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_size, rnn_output_size // 2),
            nn.Tanh(),
            nn.Linear(rnn_output_size // 2, 1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)
        conv_features = conv_features.squeeze(2).transpose(1, 2)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)
        
        # Compute attention weights
        attention_scores = self.attention(rnn_output)  # (batch_size, time_frames, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, time_frames, 1)
        
        # Apply attention
        attended_output = torch.sum(rnn_output * attention_weights, dim=1)  # (batch_size, rnn_output_size)
        
        # Classification
        logits = self.classifier(attended_output)
        
        return logits, attention_weights.squeeze(2)
