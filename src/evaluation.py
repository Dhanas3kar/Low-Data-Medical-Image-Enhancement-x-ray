"""
Evaluation utilities for model assessment and advanced loss functions with robust error handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    """Perceptual loss based on edge preservation features."""
    
    def __init__(self, use_mse: bool = True, edge_weight: float = 0.1):
        super(PerceptualLoss, self).__init__()
        if not isinstance(use_mse, bool):
            raise TypeError(f"use_mse must be bool, got {type(use_mse)}")
        if not isinstance(edge_weight, (int, float)) or edge_weight < 0:
            raise ValueError(f"edge_weight must be non-negative, got {edge_weight}")
        
        self.use_mse = use_mse
        self.edge_weight = edge_weight
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss based on edge preservation.
        
        Args:
            x: Predicted tensor
            y: Target tensor
            
        Returns:
            Loss value
        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Both x and y must be torch tensors")
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        try:
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
                return mse_loss + self.edge_weight * grad_loss
            return grad_loss
        except Exception as e:
            logger.error(f"Perceptual loss computation failed: {str(e)}")
            return torch.tensor(0.0, device=x.device, requires_grad=True)


class SSIMLoss(nn.Module):
    """SSIM-based loss function with robust computation."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super(SSIMLoss, self).__init__()
        
        if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
            raise ValueError(f"window_size must be odd int >= 3, got {window_size}")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        
        self.window_size = window_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        try:
            x = torch.arange(window_size).float()
            x = x - (window_size - 1) / 2.0
            gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            kernel = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
            kernel = kernel / kernel.sum()
            
            self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        except Exception as e:
            logger.error(f"Failed to create Gaussian kernel: {str(e)}")
            raise
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM) with error handling.
        
        Args:
            x: Predicted tensor
            y: Target tensor
            data_range: Data range for normalization
            
        Returns:
            SSIM loss value
        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Both x and y must be torch tensors")
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        if not isinstance(data_range, (int, float)) or data_range <= 0:
            raise ValueError(f"data_range must be positive, got {data_range}")
        
        try:
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
            
            # SSIM formula with numerical stability
            c1 = (0.01 * data_range) ** 2
            c2 = (0.03 * data_range) ** 2
            
            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
            
            loss = 1.0 - ssim_map.mean()
            return loss
        except Exception as e:
            logger.error(f"SSIM loss computation failed: {str(e)}")
            return torch.tensor(1.0, device=x.device, requires_grad=True)


class ModelEvaluator:
    """Evaluate model performance with multiple metrics and error handling."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        logger.info(f"ModelEvaluator initialized on {self.device}")
    
    def evaluate_batch(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions against targets for a batch with error handling.
        
        Args:
            predictions: Batch of predictions
            targets: Batch of targets
            
        Returns:
            Dictionary of metrics
        """
        if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
            raise TypeError("Both predictions and targets must be numpy arrays")
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(f"Batch size mismatch: {predictions.shape[0]} vs {targets.shape[0]}")
        
        metrics = {
            'psnr_batch': [],
            'ssim_batch': [],
            'mse_batch': [],
        }
        
        for i in range(predictions.shape[0]):
            try:
                pred = predictions[i]
                target = targets[i]
                
                # Handle different shapes
                if pred.ndim == 3:
                    pred = pred[0]
                if target.ndim == 3:
                    target = target[0]
                
                # Normalize to [0, 1]
                pred = np.clip(pred, 0, 1)
                target = np.clip(target, 0, 1)
                
                # Calculate metrics
                try:
                    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
                    metrics['psnr_batch'].append(psnr)
                except Exception as e:
                    logger.warning(f"PSNR calculation failed for sample {i}: {str(e)}")
                    metrics['psnr_batch'].append(0.0)
                
                try:
                    ssim = structural_similarity(target, pred, data_range=1.0)
                    metrics['ssim_batch'].append(ssim)
                except Exception as e:
                    logger.warning(f"SSIM calculation failed for sample {i}: {str(e)}")
                    metrics['ssim_batch'].append(0.0)
                
                mse = np.mean((target - pred) ** 2)
                metrics['mse_batch'].append(mse)
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {str(e)}")
                metrics['psnr_batch'].append(0.0)
                metrics['ssim_batch'].append(0.0)
                metrics['mse_batch'].append(1.0)
        
        # Average across batch
        result = {
            'psnr': np.mean(metrics['psnr_batch']) if metrics['psnr_batch'] else 0.0,
            'ssim': np.mean(metrics['ssim_batch']) if metrics['ssim_batch'] else 0.0,
            'mse': np.mean(metrics['mse_batch']) if metrics['mse_batch'] else 1.0,
        }
        
        return result
    
    def evaluate_dataset(self, model: nn.Module, dataloader,
                        criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluate model on entire dataset with comprehensive error handling.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader
            criterion: Optional loss criterion
            
        Returns:
            Dictionary of averaged metrics
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")
        
        model.eval()
        metrics = {'loss': [], 'psnr': [], 'ssim': [], 'mse': []}
        
        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(dataloader):
                    try:
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            low_quality, high_quality = batch_data
                        else:
                            raise ValueError("Dataloader must return (low_quality, high_quality) tuples")
                        
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
                        metrics['psnr'].append(batch_metrics['psnr'])
                        metrics['ssim'].append(batch_metrics['ssim'])
                        metrics['mse'].append(batch_metrics['mse'])
                    except Exception as e:
                        logger.error(f"Error evaluating batch {batch_idx}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Dataset evaluation failed: {str(e)}")
        
        # Average all metrics
        result = {}
        for key, values in metrics.items():
            if values:
                result[key] = np.mean(values)
        
        return result


class EarlyStopping:
    """Early stopping callback with robust state management."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 restore_best_weights: bool = True):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        if not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be positive int, got {patience}")
        if not isinstance(min_delta, (int, float)) or min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        logger.info(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}")
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if not isinstance(val_loss, (int, float)):
            raise TypeError(f"val_loss must be numeric, got {type(val_loss)}")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")
        
        try:
            if self.best_loss is None:
                self.best_loss = val_loss
                if self.restore_best_weights:
                    self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                return False
            
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                logger.info(f"EarlyStopping: Validation loss improved to {val_loss:.6f}")
                return False
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info(f"EarlyStopping: Stopped at epoch with counter={self.counter}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error in EarlyStopping: {str(e)}")
            return False
    
    def restore(self, model: nn.Module) -> None:
        """
        Restore best weights to model.
        
        Args:
            model: Model to restore weights to
        """
        if self.best_weights is None:
            logger.warning("No best weights to restore")
            return
        
        try:
            model.load_state_dict(self.best_weights)
            logger.info("Best weights restored to model")
        except Exception as e:
            logger.error(f"Failed to restore weights: {str(e)}")
            raise


def compute_metrics_over_dataset(model: nn.Module, dataloader,
                                 device: torch.device) -> Dict[str, float]:
    """
    Compute comprehensive metrics over dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    try:
        evaluator = ModelEvaluator(device=device)
        metrics = evaluator.evaluate_dataset(model, dataloader)
        return metrics
    except Exception as e:
        logger.error(f"Failed to compute metrics: {str(e)}")
        raise
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
