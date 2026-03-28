"""
Utility functions for image processing, metrics, and visualization with robust error handling.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Tuple, List, Optional, Dict, Union
import cv2
import logging
import json

logger = logging.getLogger(__name__)


class ImageMetrics:
    """Calculate image quality metrics with error handling."""
    
    @staticmethod
    def _validate_images(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Validate image arrays before metric calculation."""
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("Both y_true and y_pred must be numpy arrays")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("Images cannot be empty")
        if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
            raise ValueError("Images contain NaN or infinite values")
    
    @staticmethod
    def psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = 1.0) -> float:
        """
        Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            y_true: Ground truth image
            y_pred: Predicted image
            data_range: Data range (default 1.0 for normalized images)
            
        Returns:
            PSNR value
        """
        try:
            ImageMetrics._validate_images(y_true, y_pred)
            if not isinstance(data_range, (int, float)) or data_range <= 0:
                raise ValueError(f"data_range must be positive, got {data_range}")
            return peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
        except Exception as e:
            logger.error(f"PSNR calculation failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def ssim(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = 1.0) -> float:
        """
        Structural Similarity Index (SSIM).
        
        Args:
            y_true: Ground truth image
            y_pred: Predicted image
            data_range: Data range (default 1.0)
            
        Returns:
            SSIM value
        """
        try:
            ImageMetrics._validate_images(y_true, y_pred)
            if not isinstance(data_range, (int, float)) or data_range <= 0:
                raise ValueError(f"data_range must be positive, got {data_range}")
            return structural_similarity(y_true, y_pred, data_range=data_range)
        except Exception as e:
            logger.error(f"SSIM calculation failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error with error handling."""
        try:
            ImageMetrics._validate_images(y_true, y_pred)
            return np.mean((y_true - y_pred) ** 2)
        except Exception as e:
            logger.error(f"MSE calculation failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error with error handling."""
        try:
            ImageMetrics._validate_images(y_true, y_pred)
            return np.mean(np.abs(y_true - y_pred))
        except Exception as e:
            logger.error(f"MAE calculation failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error with error handling."""
        try:
            ImageMetrics._validate_images(y_true, y_pred)
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        except Exception as e:
            logger.error(f"RMSE calculation failed: {str(e)}")
            return 0.0


class Visualizer:
    """Visualization utilities with robust error handling."""
    
    @staticmethod
    def plot_training_history(train_losses: List[float], val_losses: List[float],
                            save_path: Optional[str] = None) -> None:
        """
        Plot training and validation losses.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Path to save figure
        """
        try:
            if not train_losses or not val_losses:
                raise ValueError("Loss lists cannot be empty")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
            ax.plot(val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=3)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training History', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Training history saved: {save_path}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot training history: {str(e)}")
            plt.close()
    
    @staticmethod
    def plot_enhancement_results(low_quality: np.ndarray, enhanced: np.ndarray,
                               high_quality: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot enhancement comparison with error handling.
        
        Args:
            low_quality: Low quality input image
            enhanced: Enhanced output image
            high_quality: Optional reference image
            save_path: Path to save figure
        """
        try:
            num_plots = 3 if high_quality is not None else 2
            fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
            if num_plots == 2:
                axes = [axes[0], axes[1]]  # Ensure consistent handling
            
            # Ensure single channel display
            def prepare_image(img):
                if len(img.shape) == 4:
                    return img[0, 0]
                elif len(img.shape) == 3:
                    return img[0]
                return img
            
            low_quality = prepare_image(low_quality)
            enhanced = prepare_image(enhanced)
            
            axes[0].imshow(low_quality, cmap='gray')
            axes[0].set_title('Input (Low Quality)')
            axes[0].axis('off')
            
            axes[1].imshow(enhanced, cmap='gray')
            axes[1].set_title('Enhanced Output')
            axes[1].axis('off')
            
            if high_quality is not None:
                high_quality = prepare_image(high_quality)
                axes[2].imshow(high_quality, cmap='gray')
                axes[2].set_title('Reference (High Quality)')
                axes[2].axis('off')
            
            plt.tight_layout()
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Enhancement results saved: {save_path}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot enhancement results: {str(e)}")
            plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, float], save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple metrics with error handling.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            save_path: Path to save figure
        """
        try:
            if not metrics_dict:
                raise ValueError("Metrics dictionary cannot be empty")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            names = list(metrics_dict.keys())
            values = list(metrics_dict.values())
            
            bars = ax.bar(names, values, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Image Quality Metrics', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Metrics comparison saved: {save_path}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot metrics: {str(e)}")
            plt.close()


class ImageProcessor:
    """Image processing utilities with robust error handling."""
    
    @staticmethod
    def normalize(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Normalize image to specified range with validation.
        
        Args:
            image: Input image array
            min_val: Minimum value of output range
            max_val: Maximum value of output range
            
        Returns:
            Normalized image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if image.size == 0:
            raise ValueError("Image is empty")
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            logger.warning("Image contained NaN/inf values - converted to 0/1")
        
        image_min = image.min()
        image_max = image.max()
        
        if image_max == image_min:
            return np.ones_like(image) * min_val
        
        normalized = (image - image_min) / (image_max - image_min)
        return normalized * (max_val - min_val) + min_val
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, std: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to image with validation.
        
        Args:
            image: Input image
            std: Standard deviation of noise
            
        Returns:
            Noisy image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if not isinstance(std, (int, float)) or std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        
        noise = np.random.normal(0, std, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        return noisy
    
    @staticmethod
    def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Add Poisson noise to image with validation.
        
        Args:
            image: Input image
            scale: Scale factor for Poisson distribution
            
        Returns:
            Noisy image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        
        image = image.copy()
        if image.min() < 0:
            image = image - image.min()
        
        scaled = image * scale
        poisson_noise = np.random.poisson(scaled) / scale
        noisy = np.clip(poisson_noise, 0, 1)
        return noisy
    
    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image using OpenCV with validation.
        
        Args:
            image: Input image
            size: Target size (height, width)
            
        Returns:
            Resized image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError(f"size must be tuple of 2 ints, got {size}")
        if any(s <= 0 for s in size):
            raise ValueError(f"size values must be positive, got {size}")
        
        try:
            if len(image.shape) == 3:
                # Handle single channel with batch dimension
                resized = cv2.resize(image[0], size, interpolation=cv2.INTER_AREA)
                return np.expand_dims(resized, 0)
            elif len(image.shape) == 2:
                return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            else:
                raise ValueError(f"Expected 2D or 3D array, got {len(image.shape)}D")
        except Exception as e:
            raise RuntimeError(f"Resize failed: {str(e)}")
    
    @staticmethod
    def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Center crop image with validation.
        
        Args:
            image: Input image
            crop_size: Crop size (height, width)
            
        Returns:
            Cropped image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if not isinstance(crop_size, tuple) or len(crop_size) != 2:
            raise ValueError(f"crop_size must be tuple of 2 ints, got {crop_size}")
        
        h, w = image.shape[-2:]
        ch, cw = crop_size
        
        if ch > h or cw > w:
            raise ValueError(f"Crop size ({ch}, {cw}) exceeds image size ({h}, {w})")
        
        start_h = (h - ch) // 2
        start_w = (w - cw) // 2
        
        if len(image.shape) == 2:
            return image[start_h:start_h+ch, start_w:start_w+cw]
        elif len(image.shape) == 3:
            return image[:, start_h:start_h+ch, start_w:start_w+cw]
        elif len(image.shape) == 4:
            return image[:, :, start_h:start_h+ch, start_w:start_w+cw]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")


class DirManager:
    """Directory management utilities with error handling."""
    
    @staticmethod
    def create_dirs(paths: Dict[str, str]) -> None:
        """
        Create multiple directories if they don't exist.
        
        Args:
            paths: Dictionary of directory names and paths
        """
        if not isinstance(paths, dict):
            raise TypeError(f"paths must be dict, got {type(paths)}")
        
        for name, path in paths.items():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ready: {name} -> {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {name} ({path}): {str(e)}")
                raise
    
    @staticmethod
    def save_metrics(metrics: Dict[str, float], save_path: str) -> None:
        """
        Save metrics to text file with error handling.
        
        Args:
            metrics: Dictionary of metric names and values
            save_path: Path to save metrics
        """
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.6f}\n")
            logger.info(f"Metrics saved: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise
    
    @staticmethod
    def save_config(config: Dict, save_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Config saved: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
            raise


def get_model_summary(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get model parameter statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable,
    }


def print_model_summary(model: torch.nn.Module) -> None:
    """Print model parameter statistics."""
    try:
        summary = get_model_summary(model)
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print(f"Total Parameters:        {summary['total_parameters']:>12,}")
        print(f"Trainable Parameters:    {summary['trainable_parameters']:>12,}")
        print(f"Non-trainable Parameters:{summary['non_trainable_parameters']:>12,}")
        print("="*50 + "\n")
    except Exception as e:
        logger.error(f"Failed to print model summary: {str(e)}")
