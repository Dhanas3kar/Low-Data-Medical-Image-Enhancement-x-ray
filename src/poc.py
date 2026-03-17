"""
Proof of Concept: Low-Data Medical Image Enhancement
Demonstrates synthetic X-ray enhancement using a lightweight UNet model.
Enhanced version with comprehensive training utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Handle imports for both direct execution and package import
try:
    from .model import UNet
    from .train import Trainer, TrainingConfig
    from .utils import Visualizer, ImageProcessor, DirManager
except ImportError:
    from model import UNet
    from train import Trainer, TrainingConfig
    from utils import Visualizer, ImageProcessor, DirManager

import yaml


class SyntheticXrayDataset(Dataset):
    """Generate synthetic low-quality and high-quality X-ray pairs for POC."""
    
    def __init__(self, num_samples=20, size=256, is_train=True, noise_level=0.2):
        self.num_samples = num_samples
        self.size = size
        self.is_train = is_train
        self.noise_level = noise_level
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic low-quality (noisy) image
        low_quality = self._generate_xray_image(noise_level=0.3)
        # Generate corresponding high-quality reference
        high_quality = self._generate_xray_image(noise_level=0.05)
        
        return torch.FloatTensor(low_quality), torch.FloatTensor(high_quality)
    
    def _generate_xray_image(self, noise_level=0.2):
        """Generate synthetic X-ray-like image."""
        img = np.zeros((1, self.size, self.size))
        
        # Create simulated anatomy (circles, gradients, variations)
        num_objects = np.random.randint(2, 5)
        for _ in range(num_objects):
            y, x = np.random.randint(50, self.size-50, 2)
            r = np.random.randint(20, 80)
            Y, X = np.ogrid[:self.size, :self.size]
            circle_mask = (X - x)**2 + (Y - y)**2 <= r**2
            intensity = np.random.rand() * 0.6 + 0.3
            img[0][circle_mask] += intensity
        
        # Add gradient for anatomy depth
        gradient = np.linspace(0, 0.3, self.size)
        img[0] += gradient[np.newaxis, :] * 0.2
        
        # Add subtle texture
        texture = np.random.rand(self.size, self.size) * 0.1
        img[0] += texture
        
        # Add noise
        noise = np.random.normal(0, noise_level, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        return img


def main():
    print("=" * 70)
    print("LOW-DATA MEDICAL IMAGE ENHANCEMENT - PROOF OF CONCEPT")
    print("=" * 70)
    
    # Create directories
    paths = {
        'models': 'models',
        'outputs': 'outputs',
        'logs': 'logs',
    }
    DirManager.create_dirs(paths)
    
    # Load/verify config
    config = TrainingConfig('configs/config.yaml')
    print(f"\n✓ Configuration loaded")
    print(f"  Device: cuda" if torch.cuda.is_available() else f"  Device: cpu")
    print(f"  Model: {config.model.get('architecture', 'UNet')}")
    print(f"  Epochs: {config.training['epochs']}")
    print(f"  Batch Size: {config.data['batch_size']}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create synthetic dataset
    print(f"\n✓ Generating synthetic X-ray dataset...")
    train_dataset = SyntheticXrayDataset(
        num_samples=100,
        size=config.data['image_size'],
        is_train=True
    )
    val_dataset = SyntheticXrayDataset(
        num_samples=20,
        size=config.data['image_size'],
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data['batch_size'],
        num_workers=0
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Train model
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    trainer.train(train_loader, val_loader)
    
    # Demonstration
    print(f"\n{'='*70}")
    print("ENHANCEMENT DEMONSTRATION")
    print(f"{'='*70}\n")
    
    # Generate sample enhancement
    sample_dataset = SyntheticXrayDataset(num_samples=1)
    sample_low, sample_high = sample_dataset[0]
    
    enhanced = trainer.infer(sample_low)
    
    # Visualize results
    print("✓ Creating visualization...")
    Visualizer.plot_enhancement_results(
        sample_low.numpy(),
        enhanced,
        sample_high.numpy(),
        save_path='outputs/poc_enhancement_result.png'
    )
    
    # Plot training history
    Visualizer.plot_training_history(
        trainer.history['train_loss'],
        trainer.history['val_loss'],
        save_path='outputs/poc_training_history.png'
    )
    
    print(f"\n{'='*70}")
    print("PROOF OF CONCEPT COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Model saved: models/model_final.pt")
    print(f"✓ Results saved to: outputs/")
    print(f"✓ Logs saved to: logs/")
    
    # Print final stats
    print(f"\nFinal Metrics:")
    if trainer.history['val_metrics']:
        last_metrics = trainer.history['val_metrics'][-1]
        print(f"  Final Val Loss: {trainer.history['val_loss'][-1]:.6f}")
        if 'psnr' in last_metrics:
            print(f"  Final PSNR: {last_metrics.get('psnr', 0):.4f}")
        if 'ssim' in last_metrics:
            print(f"  Final SSIM: {last_metrics.get('ssim', 0):.4f}")


if __name__ == "__main__":
    main()

