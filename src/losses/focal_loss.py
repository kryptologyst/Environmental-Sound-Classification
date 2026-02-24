"""Loss functions for audio classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,) or one-hot of shape (batch_size, num_classes)
            
        Returns:
            Focal loss value
        """
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        else:
            targets_one_hot = targets
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets_one_hot, reduction="none")
        
        # Compute p_t
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Label smoothing cross entropy loss
        """
        num_classes = inputs.size(1)
        log_preds = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smoothed_targets = torch.zeros_like(log_preds)
        smoothed_targets.fill_(self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        loss = -smoothed_targets * log_preds
        loss = loss.sum(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MixUpLoss(nn.Module):
    """
    Loss function for MixUp augmentation.
    
    Args:
        criterion: Base loss function
    """
    
    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MixUp loss.
        
        Args:
            inputs: Predicted logits
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixing weights
            
        Returns:
            MixUp loss value
        """
        return lam * self.criterion(inputs, targets_a) + (1 - lam) * self.criterion(inputs, targets_b)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for representation learning.
    
    Args:
        margin: Margin for contrastive loss
        temperature: Temperature parameter
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Feature embeddings of shape (batch_size, embedding_dim)
            labels: Ground truth labels of shape (batch_size,)
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive and negative masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_ne = ~labels_eq
        
        # Remove diagonal (self-similarity)
        labels_eq.fill_diagonal_(False)
        
        # Compute positive and negative similarities
        pos_sim = similarity_matrix[labels_eq]
        neg_sim = similarity_matrix[labels_ne]
        
        if len(pos_sim) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute contrastive loss
        pos_loss = -pos_sim.mean()
        neg_loss = F.relu(self.margin - neg_sim).mean()
        
        return pos_loss + neg_loss


class CenterLoss(nn.Module):
    """
    Center loss for feature learning.
    
    Args:
        num_classes: Number of classes
        feature_dim: Dimension of features
        lambda_c: Weight for center loss
    """
    
    def __init__(self, num_classes: int, feature_dim: int, lambda_c: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c
        
        # Initialize centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            features: Feature embeddings of shape (batch_size, feature_dim)
            labels: Ground truth labels of shape (batch_size,)
            
        Returns:
            Center loss value
        """
        batch_size = features.size(0)
        
        # Get centers for current batch
        centers_batch = self.centers[labels]
        
        # Compute center loss
        center_loss = F.mse_loss(features, centers_batch)
        
        return self.lambda_c * center_loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    loss_functions = {
        "cross_entropy": nn.CrossEntropyLoss,
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingCrossEntropy,
        "contrastive": ContrastiveLoss,
        "center": CenterLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
