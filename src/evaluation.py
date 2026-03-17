"""
Evaluation utilities for model assessment and advanced loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter


class PerceptualLoss(nn.Module):
    """Perceptual loss based on pre-trained features."""
    
    def __init__(self, use_mse: bool = True):
        super(PerceptualLoss, self).__init__()
        self.use_mse = use_mse
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # For low-data scenarios, use simple edge-based perceptual loss
        # Compute gradients to measure edge preservation
        x_grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        x_grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        y_grad_x = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        y_grad_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        
        # Compare gradients
        grad_loss = F.l1_loss(x_grad_x, y_grad_x) + F.l1_loss(x_grad_y, y_grad_y)
        
        # Combine with MSE if specified
        if self.use_mse:
            mse_loss = F.mse_loss(x, y)
            return mse_loss + 0.1 * grad_loss
        return grad_loss


class SSIMLoss(nn.Module):
    """SSIM-based loss function."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        x = torch.arange(window_size).float()
        x = x - (window_size - 1) / 2.0
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        kernel = kernel / kernel.sum()
        
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        # Ensure grayscale
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if y.shape[1] == 3:
            y = y.mean(dim=1, keepdim=True)
        
        # Pad for convolution
        pad = self.window_size // 2
        
        # Compute local means
        mu1 = F.conv2d(x, self.kernel, padding=pad)
        mu2 = F.conv2d(y, self.kernel, padding=pad)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances
        sigma1_sq = F.conv2d(x ** 2, self.kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(y ** 2, self.kernel, padding=pad) - mu2_sq
        sigma12 = F.conv2d(x * y, self.kernel, padding=pad) - mu1_mu2
        
        # SSIM formula
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return 1.0 - ssim_map.mean()


class ModelEvaluator:
    """Evaluate model performance with multiple metrics."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
    
    def evaluate_batch(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions against targets for a batch."""
        metrics = {}
        
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            target = targets[i]
            
            # Handle different shapes
            if pred.ndim == 3:
                pred = pred[0]  # Remove channel dimension
            if target.ndim == 3:
                target = target[0]
            
            # Normalize to [0, 1]
            pred = np.clip(pred, 0, 1)
            target = np.clip(target, 0, 1)
            
            # Calculate metrics
            if f'psnr_batch' not in metrics:
                metrics['psnr_batch'] = []
                metrics['ssim_batch'] = []
                metrics['mse_batch'] = []
            
            try:
                psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
                metrics['psnr_batch'].append(psnr)
            except:
                metrics['psnr_batch'].append(0.0)
            
            try:
                ssim = structural_similarity(target, pred, data_range=1.0)
                metrics['ssim_batch'].append(ssim)
            except:
                metrics['ssim_batch'].append(0.0)
            
            mse = np.mean((target - pred) ** 2)
            metrics['mse_batch'].append(mse)
        
        # Average across batch
        result = {
            'psnr': np.mean(metrics['psnr_batch']),
            'ssim': np.mean(metrics['ssim_batch']),
            'mse': np.mean(metrics['mse_batch']),
        }
        
        return result
    
    def evaluate_dataset(self, model: torch.nn.Module, dataloader, 
                        criterion=None) -> Dict[str, float]:
        """Evaluate model on entire dataset."""
        model.eval()
        metrics = {'loss': [], 'psnr': [], 'ssim': [], 'mse': []}
        
        with torch.no_grad():
            for low_quality, high_quality in dataloader:
                low_quality = low_quality.to(self.device)
                high_quality = high_quality.to(self.device)
                
                # Forward pass
                output = model(low_quality)
                
                # Loss
                if criterion:
                    loss = criterion(output, high_quality)
                    metrics['loss'].append(loss.item())
                
                # Convert to numpy
                pred = output.cpu().numpy()
                target = high_quality.cpu().numpy()
                
                # Image metrics
                batch_metrics = self.evaluate_batch(pred, target)
                metrics['psnr'].extend([batch_metrics['psnr']])
                metrics['ssim'].extend([batch_metrics['ssim']])
                metrics['mse'].extend([batch_metrics['mse']])
        
        # Average all metrics
        result = {}
        for key, values in metrics.items():
            if values:
                result[key] = np.mean(values)
        
        return result


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        Returns True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Should stop
        
        return False  # Continue training
    
    def restore(self, model: torch.nn.Module):
        """Restore best weights to model."""
        if self.best_weights:
            model.load_state_dict(self.best_weights)


def compute_metrics_over_dataset(model: torch.nn.Module, dataloader,
                                 device: torch.device) -> Dict[str, float]:
    """Compute comprehensive metrics over dataset."""
    evaluator = ModelEvaluator(device=device)
    metrics = evaluator.evaluate_dataset(model, dataloader)
    return metrics
