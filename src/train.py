"""
Advanced training script with callbacks, metrics, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yaml
import json
from datetime import datetime
from tqdm import tqdm

from .model import UNet
from .evaluation import EarlyStopping, ModelEvaluator, PerceptualLoss, SSIMLoss
from .utils import print_model_summary, get_model_summary, DirManager, Visualizer


class TrainingConfig:
    """Training configuration container."""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.__dict__.update(config_dict)
    
    def save(self, save_path: str):
        """Save config to file."""
        with open(save_path, 'w') as f:
            yaml.dump(self.__dict__, f)


class Trainer:
    """Main training class with callbacks and monitoring."""
    
    def __init__(self, config: TrainingConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        DirManager.create_dirs(self.config.paths)
        
        # Initialize model
        self.model = UNet(
            in_channels=self.config.model.get('in_channels', 1),
            out_channels=self.config.model.get('out_channels', 1),
            depth=self.config.model.get('depth', 3),
            dropout=self.config.model.get('dropout', 0.2)
        ).to(self.device)
        
        # Print model summary
        print_model_summary(self.model)
        
        # Initialize loss function
        self.criterion = self._get_criterion()
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.training.get('lr_step_size', 20),
            gamma=self.config.training.get('lr_gamma', 0.5)
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.get('early_stopping_patience', 10),
            restore_best_weights=True
        )
        
        # Evaluator
        self.evaluator = ModelEvaluator(device=self.device)
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rate': []
        }
        
        # Timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_criterion(self) -> nn.Module:
        """Get loss criterion based on config."""
        loss_name = self.config.training.get('loss_function', 'MSE').lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'l1':
            return nn.L1Loss()
        elif loss_name == 'ssim':
            return SSIMLoss()
        elif loss_name == 'perceptual':
            return PerceptualLoss(use_mse=True)
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on config."""
        opt_name = self.config.training.get('optimizer', 'Adam').lower()
        lr = self.config.training.get('learning_rate', 0.001)
        weight_decay = self.config.regularization.get('weight_decay', 0.0001)
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_epoch(self, train_loader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for low_quality, high_quality in pbar:
            low_quality = low_quality.to(self.device)
            high_quality = high_quality.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(low_quality)
            loss = self.criterion(output, high_quality)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for low_quality, high_quality in val_loader:
                low_quality = low_quality.to(self.device)
                high_quality = high_quality.to(self.device)
                
                output = self.model(low_quality)
                loss = self.criterion(output, high_quality)
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        
        # Calculate additional metrics
        metrics = self.evaluator.evaluate_dataset(self.model, val_loader, self.criterion)
        
        return val_loss, metrics
    
    def train(self, train_loader, val_loader):
        """Complete training loop."""
        print("\n" + "=" * 70)
        print("TRAINING START")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training['epochs']}")
        print(f"Loss Function: {self.config.training.get('loss_function', 'MSE')}")
        print(f"Optimizer: {self.config.training.get('optimizer', 'Adam')}")
        print("=" * 70 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.training['epochs'] + 1):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            if epoch % self.config.training.get('print_interval', 1) == 0:
                print(f"Epoch {epoch:3d}/{self.config.training['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if 'psnr' in val_metrics:
                    print(f"              PSNR: {val_metrics.get('psnr', 0):.4f} | "
                          f"SSIM: {val_metrics.get('ssim', 0):.4f}")
            
            # Save checkpoint
            if epoch % self.config.training.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping at epoch {epoch}")
                self.early_stopping.restore(self.model)
                break
            
            # Track best loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70 + "\n")
        
        # Save final model
        self.save_model('model_final')
        
        # Save training history
        self.save_history()
        
        # Generate plots
        self.plot_training_history()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        ckpt_path = Path(self.config.paths['model_dir']) / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    
    def save_model(self, name: str = 'model'):
        """Save trained model."""
        model_path = Path(self.config.paths['model_dir']) / f'{name}.pt'
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
    
    def save_history(self):
        """Save training history."""
        history_path = Path(self.config.paths['logs_dir']) / f'history_{self.timestamp}.json'
        
        # Convert numpy values to float for JSON serialization
        history_serializable = {}
        for key, values in self.history.items():
            if isinstance(values[0], (int, float)):
                history_serializable[key] = [float(v) for v in values]
            elif isinstance(values[0], dict):
                history_serializable[key] = values
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved: {history_path}")
    
    def plot_training_history(self):
        """Plot and save training history plots."""
        output_path = Path(self.config.paths['output_dir']) / f'training_plot_{self.timestamp}.png'
        Visualizer.plot_training_history(
            self.history['train_loss'],
            self.history['val_loss'],
            save_path=str(output_path)
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from: {checkpoint_path}")
    
    def load_model(self, model_path: str):
        """Load trained model."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Loaded model from: {model_path}")
    
    def infer(self, input_image: torch.Tensor) -> np.ndarray:
        """Run inference on single image."""
        self.model.eval()
        with torch.no_grad():
            if len(input_image.shape) == 2:
                input_image = input_image[None, None, :, :]  # Add batch and channel dims
            elif len(input_image.shape) == 3:
                input_image = input_image[None, :, :, :]  # Add batch dim
            
            input_tensor = input_image.to(self.device)
            output = self.model(input_tensor)
            return output.cpu().numpy()
