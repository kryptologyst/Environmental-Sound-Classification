"""Evaluation module for environmental sound classification."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device
from ..metrics.classification import compute_all_metrics


class Evaluator:
    """
    Evaluator class for environmental sound classification.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to use for evaluation
        save_dir: Directory to save evaluation results
        class_names: Names of classes
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: Union[str, torch.device] = "auto",
        save_dir: Union[str, Path] = "assets",
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = get_device(device)
        self.save_dir = Path(save_dir)
        self.class_names = class_names or [f"Class_{i}" for i in range(50)]
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, save_predictions: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        print("Evaluating model on test data...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_filenames = []
        all_categories = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluation")
            
            for batch in pbar:
                # Move batch to device
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_filenames.extend(batch["filename"])
                all_categories.extend(batch["category"])
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Compute metrics
        print("Computing evaluation metrics...")
        metrics = compute_all_metrics(
            predictions=predictions,
            targets=labels,
            probabilities=probabilities,
            num_classes=len(self.class_names),
            class_names=self.class_names,
        )
        
        # Create results dictionary
        results = {
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
            "predictions": predictions,
            "labels": labels,
            "probabilities": probabilities,
            "filenames": all_filenames,
            "categories": all_categories,
        }
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(results)
        
        # Save metrics
        self._save_metrics(results["metrics"])
        
        # Create visualizations
        self._create_visualizations(results)
        
        # Print summary
        self._print_summary(results["metrics"])
        
        return results
    
    def _save_predictions(self, results: Dict[str, Any]) -> None:
        """Save predictions to CSV file."""
        predictions_df = pd.DataFrame({
            "filename": results["filenames"],
            "true_category": results["categories"],
            "true_label": results["labels"],
            "predicted_label": results["predictions"],
            "predicted_category": [self.class_names[i] for i in results["predictions"]],
            "confidence": np.max(results["probabilities"], axis=1),
        })
        
        # Add top-5 predictions
        top5_indices = np.argsort(results["probabilities"], axis=1)[:, -5:]
        top5_categories = [[self.class_names[i] for i in indices] for indices in top5_indices]
        predictions_df["top5_categories"] = top5_categories
        
        predictions_file = self.save_dir / "predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions saved to {predictions_file}")
    
    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save metrics to JSON file."""
        metrics_file = self.save_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    def _create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create visualization plots."""
        # Confusion matrix
        self._plot_confusion_matrix(results["confusion_matrix"])
        
        # Per-class accuracy
        self._plot_per_class_accuracy(results["confusion_matrix"])
        
        # Confidence distribution
        self._plot_confidence_distribution(results["probabilities"], results["labels"])
        
        # Top-5 accuracy
        self._plot_top5_accuracy(results["probabilities"], results["labels"])
    
    def _plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Normalized Count"}
        )
        
        plt.title("Confusion Matrix (Normalized)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_file = self.save_dir / "confusion_matrix.png"
        plt.savefig(confusion_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {confusion_file}")
    
    def _plot_per_class_accuracy(self, confusion_matrix: np.ndarray) -> None:
        """Plot per-class accuracy."""
        per_class_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(self.class_names)), per_class_acc)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy")
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha="right")
        plt.ylim(0, 1)
        
        # Color bars based on accuracy
        for bar, acc in zip(bars, per_class_acc):
            if acc >= 0.8:
                bar.set_color("green")
            elif acc >= 0.6:
                bar.set_color("orange")
            else:
                bar.set_color("red")
        
        plt.tight_layout()
        
        per_class_file = self.save_dir / "per_class_accuracy.png"
        plt.savefig(per_class_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Per-class accuracy plot saved to {per_class_file}")
    
    def _plot_confidence_distribution(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        """Plot confidence distribution for correct and incorrect predictions."""
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        correct = predictions == labels
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(
            max_probs[correct],
            bins=50,
            alpha=0.7,
            label="Correct Predictions",
            color="green",
            density=True
        )
        plt.hist(
            max_probs[~correct],
            bins=50,
            alpha=0.7,
            label="Incorrect Predictions",
            color="red",
            density=True
        )
        
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.title("Confidence Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        confidence_file = self.save_dir / "confidence_distribution.png"
        plt.savefig(confidence_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confidence distribution plot saved to {confidence_file}")
    
    def _plot_top5_accuracy(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        """Plot top-K accuracy for different K values."""
        k_values = range(1, min(6, len(self.class_names) + 1))
        top_k_accuracies = []
        
        for k in k_values:
            top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
            correct = np.array([labels[i] in top_k_predictions[i] for i in range(len(labels))])
            top_k_accuracies.append(correct.mean())
        
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, top_k_accuracies, "bo-", linewidth=2, markersize=8)
        plt.xlabel("K")
        plt.ylabel("Top-K Accuracy")
        plt.title("Top-K Accuracy")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels on points
        for k, acc in zip(k_values, top_k_accuracies):
            plt.annotate(f"{acc:.3f}", (k, acc), textcoords="offset points", xytext=(0, 10), ha="center")
        
        plt.tight_layout()
        
        top_k_file = self.save_dir / "top_k_accuracy.png"
        plt.savefig(top_k_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Top-K accuracy plot saved to {top_k_file}")
    
    def _print_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Micro F1-Score: {metrics['f1_micro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print(f"Macro Precision: {metrics['precision_macro']:.4f}")
        print(f"Macro Recall: {metrics['recall_macro']:.4f}")
        
        if "top5_accuracy" in metrics:
            print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        
        if "auc_macro" in metrics:
            print(f"Macro AUC: {metrics['auc_macro']:.4f}")
        
        print("="*60)
    
    def create_leaderboard(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a leaderboard with model performance."""
        metrics = results["metrics"]
        
        leaderboard = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Macro F1-Score",
                "Micro F1-Score",
                "Weighted F1-Score",
                "Macro Precision",
                "Macro Recall",
            ],
            "Value": [
                metrics["accuracy"],
                metrics["f1_macro"],
                metrics["f1_micro"],
                metrics["f1_weighted"],
                metrics["precision_macro"],
                metrics["recall_macro"],
            ]
        })
        
        # Add optional metrics
        if "top5_accuracy" in metrics:
            leaderboard = pd.concat([
                leaderboard,
                pd.DataFrame({"Metric": ["Top-5 Accuracy"], "Value": [metrics["top5_accuracy"]]})
            ], ignore_index=True)
        
        if "auc_macro" in metrics:
            leaderboard = pd.concat([
                leaderboard,
                pd.DataFrame({"Metric": ["Macro AUC"], "Value": [metrics["auc_macro"]]})
            ], ignore_index=True)
        
        # Save leaderboard
        leaderboard_file = self.save_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_file, index=False)
        print(f"Leaderboard saved to {leaderboard_file}")
        
        return leaderboard
