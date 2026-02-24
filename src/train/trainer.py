"""Training module for environmental sound classification."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..utils.device import get_device, count_parameters, format_time
from ..metrics.classification import ClassificationMetrics
from ..losses.focal_loss import get_loss_function


class Trainer:
    """
    Trainer class for environmental sound classification.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to use for training
        save_dir: Directory to save checkpoints and logs
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        gradient_clip_val: Gradient clipping value
        accumulate_grad_batches: Number of batches to accumulate gradients
        val_check_interval: Validation check interval (in epochs)
        log_every_n_steps: Log every n steps
        save_every_n_epochs: Save checkpoint every n epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Union[str, torch.device] = "auto",
        save_dir: Union[str, Path] = "checkpoints",
        max_epochs: int = 100,
        patience: int = 15,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        val_check_interval: float = 1.0,
        log_every_n_steps: int = 100,
        save_every_n_epochs: int = 5,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = get_device(device)
        self.save_dir = Path(save_dir)
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.val_check_interval = val_check_interval
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = self.save_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Move model to device
        self.model.to(self.device)
        
        # Print model info
        num_params = count_parameters(self.model)
        print(f"Model has {num_params:,} trainable parameters")
        print(f"Training on device: {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            # Update metrics
            epoch_loss += loss.item() * self.accumulate_grad_batches
            epoch_acc += accuracy.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.accumulate_grad_batches:.4f}",
                "acc": f"{accuracy.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log metrics
            if self.global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("train/loss", loss.item() * self.accumulate_grad_batches, self.global_step)
                self.writer.add_scalar("train/accuracy", accuracy.item(), self.global_step)
                self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                # Compute probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Update metrics
                epoch_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute metrics
        avg_loss = epoch_loss / len(self.val_loader)
        
        # Classification metrics
        metrics_calculator = ClassificationMetrics(
            num_classes=len(self.train_loader.dataset.class_names),
            class_names=self.train_loader.dataset.class_names
        )
        
        metrics = metrics_calculator.compute_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
        
        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_micro": metrics["f1_micro"],
        }
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_accuracies = checkpoint["val_accuracies"]
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
    
    def train(self, resume_from: Optional[Union[str, Path]] = None) -> None:
        """Train the model."""
        start_time = time.time()
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])
            
            # Validate
            if (epoch + 1) % self.val_check_interval == 0:
                val_metrics = self.validate_epoch()
                self.val_losses.append(val_metrics["loss"])
                self.val_accuracies.append(val_metrics["accuracy"])
                
                # Log validation metrics
                self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("val/f1_macro", val_metrics["f1_macro"], epoch)
                self.writer.add_scalar("val/f1_micro", val_metrics["f1_micro"], epoch)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                is_best = val_metrics["accuracy"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(is_best)
                
                # Print epoch summary
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch + 1}/{self.max_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val F1 Macro: {val_metrics['f1_macro']:.4f}, Val F1 Micro: {val_metrics['f1_micro']:.4f}")
                print(f"  Best Val Acc: {self.best_val_acc:.4f}")
                print(f"  Elapsed Time: {format_time(elapsed_time)}")
                print(f"  Patience: {self.patience_counter}/{self.patience}")
                print("-" * 50)
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Close writer
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
