"""Evaluation metrics for audio classification."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassificationMetrics:
    """
    Classification metrics calculator.
    
    Args:
        num_classes: Number of classes
        class_names: Names of classes (optional)
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def compute_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            probabilities: Predicted class probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["precision_macro"] = precision_score(targets, predictions, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(targets, predictions, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro", zero_division=0)
        metrics["f1_micro"] = f1_score(targets, predictions, average="micro", zero_division=0)
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted", zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f"precision_{class_name}"] = precision_per_class[i]
            metrics[f"recall_{class_name}"] = recall_per_class[i]
            metrics[f"f1_{class_name}"] = f1_per_class[i]
        
        # AUC (if probabilities are provided)
        if probabilities is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics["auc"] = roc_auc_score(targets, probabilities[:, 1])
                else:
                    # Multi-class classification
                    metrics["auc_macro"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="macro"
                    )
                    metrics["auc_weighted"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="weighted"
                    )
            except ValueError:
                # Handle cases where AUC cannot be computed
                pass
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            
        Returns:
            Confusion matrix
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return confusion_matrix(targets, predictions)
    
    def compute_classification_report(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> str:
        """
        Compute detailed classification report.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            
        Returns:
            Classification report string
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return classification_report(
            targets,
            predictions,
            target_names=self.class_names,
            zero_division=0,
        )


class TopKAccuracy:
    """
    Top-K accuracy metric.
    
    Args:
        k: Number of top predictions to consider
    """
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def __call__(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute top-K accuracy.
        
        Args:
            predictions: Predicted probabilities of shape (batch_size, num_classes)
            targets: Ground truth class labels of shape (batch_size,)
            
        Returns:
            Top-K accuracy
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Get top-k predictions
        top_k_predictions = np.argsort(predictions, axis=1)[:, -self.k:]
        
        # Check if targets are in top-k predictions
        correct = np.array([targets[i] in top_k_predictions[i] for i in range(len(targets))])
        
        return correct.mean()


class AveragePrecision:
    """
    Average precision metric for multi-class classification.
    """
    
    def __call__(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute average precision.
        
        Args:
            predictions: Predicted probabilities of shape (batch_size, num_classes)
            targets: Ground truth class labels of shape (batch_size,)
            
        Returns:
            Average precision
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Convert to one-hot encoding
        targets_one_hot = np.zeros_like(predictions)
        targets_one_hot[np.arange(len(targets)), targets] = 1
        
        # Compute average precision for each class
        ap_scores = []
        for i in range(predictions.shape[1]):
            if np.sum(targets_one_hot[:, i]) > 0:  # Only if class exists
                ap_scores.append(
                    roc_auc_score(targets_one_hot[:, i], predictions[:, i])
                )
        
        return np.mean(ap_scores) if ap_scores else 0.0


def compute_all_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_classes: int = 50,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Compute all available metrics.
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth class labels
        probabilities: Predicted class probabilities (optional)
        num_classes: Number of classes
        class_names: Names of classes (optional)
        
    Returns:
        Dictionary of all metrics
    """
    metrics_calculator = ClassificationMetrics(num_classes, class_names)
    
    # Basic metrics
    metrics = metrics_calculator.compute_metrics(predictions, targets, probabilities)
    
    # Confusion matrix
    metrics["confusion_matrix"] = metrics_calculator.compute_confusion_matrix(
        predictions, targets
    )
    
    # Classification report
    metrics["classification_report"] = metrics_calculator.compute_classification_report(
        predictions, targets
    )
    
    # Top-K accuracy
    if probabilities is not None:
        top_k_acc = TopKAccuracy(k=5)
        metrics["top5_accuracy"] = top_k_acc(probabilities, targets)
        
        # Average precision
        avg_precision = AveragePrecision()
        metrics["average_precision"] = avg_precision(probabilities, targets)
    
    return metrics
