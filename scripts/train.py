#!/usr/bin/env python3
"""
Main training script for environmental sound classification.

This script trains a CRNN model on the ESC-50 dataset with proper configuration,
logging, and evaluation.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.crnn import CRNNModel
from src.data.esc50_dataset import ESC50Dataset, SyntheticDataset
from src.features.melspectrogram import MelSpectrogramExtractor
from src.features.augmentation import SpecAugment, MixUp, ComposeAugmentations
from src.losses.focal_loss import get_loss_function
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator
from src.utils.device import get_device, set_seed


def create_data_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    
    # Feature extractor
    feature_extractor = MelSpectrogramExtractor(
        sample_rate=cfg.data.sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=cfg.data.sample_rate // 2,
        power=2.0,
        normalized=True,
    )
    
    # Augmentation
    augmentations = []
    if cfg.data.augmentation.enabled:
        if cfg.data.augmentation.spec_augment:
            augmentations.append(SpecAugment(
                time_mask_param=cfg.data.augmentation.spec_augment.time_mask_param,
                freq_mask_param=cfg.data.augmentation.spec_augment.freq_mask_param,
                num_time_masks=cfg.data.augmentation.spec_augment.num_time_masks,
                num_freq_masks=cfg.data.augmentation.spec_augment.num_freq_masks,
            ))
        
        if cfg.data.augmentation.mixup:
            augmentations.append(MixUp(alpha=cfg.data.augmentation.mixup_alpha))
    
    transform = ComposeAugmentations(augmentations) if augmentations else None
    
    # Check if real dataset exists, otherwise use synthetic
    data_dir = Path(cfg.paths.data_dir)
    if not (data_dir / "raw" / "ESC-50-master").exists():
        print("ESC-50 dataset not found. Using synthetic dataset for demonstration.")
        
        # Create synthetic datasets
        train_dataset = SyntheticDataset(
            num_samples=800,
            num_classes=cfg.data.num_classes,
            sample_rate=cfg.data.sample_rate,
            duration=cfg.data.duration,
            feature_extractor=feature_extractor,
        )
        
        val_dataset = SyntheticDataset(
            num_samples=100,
            num_classes=cfg.data.num_classes,
            sample_rate=cfg.data.sample_rate,
            duration=cfg.data.duration,
            feature_extractor=feature_extractor,
        )
        
        test_dataset = SyntheticDataset(
            num_samples=100,
            num_classes=cfg.data.num_classes,
            sample_rate=cfg.data.sample_rate,
            duration=cfg.data.duration,
            feature_extractor=feature_extractor,
        )
        
        class_names = [f"class_{i}" for i in range(cfg.data.num_classes)]
    else:
        # Create real datasets
        train_dataset = ESC50Dataset(
            data_dir=data_dir,
            sample_rate=cfg.data.sample_rate,
            duration=cfg.data.duration,
            feature_extractor=feature_extractor,
            split="train",
            transform=transform,
        )
        
        val_dataset = ESC50Dataset(
            data_dir=data_dir,
            sample_rate=cfg.data.sample_rate,
            duration=cfg.data.duration,
            feature_extractor=feature_extractor,
            split="val",
        )
        
        test_dataset = ESC50Dataset(
            data_dir=data_dir,
            sample_rate=cfg.data.sample_rate,
            duration=cfg.data.duration,
            feature_extractor=feature_extractor,
            split="test",
        )
        
        class_names = train_dataset.class_names
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    return train_loader, val_loader, test_loader, class_names


def create_model(cfg: DictConfig, num_classes: int) -> nn.Module:
    """Create the model."""
    model = CRNNModel(
        input_channels=cfg.model.input_channels,
        num_classes=num_classes,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    )
    
    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 50)
    
    # Create directories
    for path_name in ["checkpoints_dir", "logs_dir", "assets_dir"]:
        Path(cfg.paths[path_name]).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(cfg)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create model
    print("Creating model...")
    model = create_model(cfg, len(class_names))
    
    # Loss function
    criterion = get_loss_function(
        cfg.training.loss._target_.split(".")[-1],
        **{k: v for k, v in cfg.training.loss.items() if k != "_target_"}
    )
    
    # Optimizer
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer,
        params=model.parameters()
    )
    
    # Scheduler
    scheduler = None
    if cfg.training.scheduler:
        scheduler = hydra.utils.instantiate(
            cfg.training.scheduler,
            optimizer=optimizer
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.device,
        save_dir=cfg.paths.checkpoints_dir,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        save_every_n_epochs=cfg.logging.save_every_n_epochs,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=cfg.device,
        save_dir=cfg.paths.assets_dir,
        class_names=class_names,
    )
    
    results = evaluator.evaluate(save_predictions=True)
    leaderboard = evaluator.create_leaderboard(results)
    
    print("Training and evaluation completed!")
    print(f"Results saved to {cfg.paths.assets_dir}")


if __name__ == "__main__":
    main()
