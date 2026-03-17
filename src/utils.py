"""
Utility functions for image processing, metrics, and visualization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Tuple, List, Optional
import cv2


class ImageMetrics:
    """Calculate image quality metrics."""
    
    @staticmethod
    def psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio (PSNR)."""
        if y_true.shape != y_pred.shape:
            raise ValueError("Images must have same shape")
        return peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
    
    @staticmethod
    def ssim(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = 1.0) -> float:
        """Structural Similarity Index (SSIM)."""
        if y_true.shape != y_pred.shape:
            raise ValueError("Images must have same shape")
        return structural_similarity(y_true, y_pred, data_range=data_range)
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_training_history(train_losses: List[float], val_losses: List[float],
                            save_path: Optional[str] = None):
        """Plot training and validation losses."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', linewidth=2)
        ax.plot(val_losses, label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {save_path}")
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_enhancement_results(low_quality: np.ndarray, enhanced: np.ndarray,
                               high_quality: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
        """Plot enhancement comparison."""
        num_plots = 3 if high_quality is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
        
        # Ensure single channel display
        if len(low_quality.shape) == 4:
            low_quality = low_quality[0, 0]
        elif len(low_quality.shape) == 3:
            low_quality = low_quality[0]
        
        if len(enhanced.shape) == 4:
            enhanced = enhanced[0, 0]
        elif len(enhanced.shape) == 3:
            enhanced = enhanced[0]
        
        axes[0].imshow(low_quality, cmap='gray')
        axes[0].set_title('Input (Low Quality)')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title('Enhanced Output')
        axes[1].axis('off')
        
        if high_quality is not None:
            if len(high_quality.shape) == 4:
                high_quality = high_quality[0, 0]
            elif len(high_quality.shape) == 3:
                high_quality = high_quality[0]
            axes[2].imshow(high_quality, cmap='gray')
            axes[2].set_title('Reference (High Quality)')
            axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved enhancement results to {save_path}")
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict, save_path: Optional[str] = None):
        """Plot comparison of multiple metrics."""
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
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved metrics comparison to {save_path}")
        else:
            plt.show()
        plt.close()


class ImageProcessor:
    """Image processing utilities."""
    
    @staticmethod
    def normalize(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """Normalize image to specified range."""
        image_min = image.min()
        image_max = image.max()
        
        if image_max == image_min:
            return np.ones_like(image) * min_val
        
        normalized = (image - image_min) / (image_max - image_min)
        return normalized * (max_val - min_val) + min_val
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, std, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        return noisy
    
    @staticmethod
    def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Add Poisson noise to image."""
        if image.min() < 0:
            image = image - image.min()
        
        scaled = image * scale
        poisson_noise = np.random.poisson(scaled) / scale
        noisy = np.clip(poisson_noise, 0, 1)
        return noisy
    
    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image using OpenCV."""
        if len(image.shape) == 3:
            # Handle batch dimension
            resized = cv2.resize(image[0], size, interpolation=cv2.INTER_AREA)
            return np.expand_dims(resized, 0)
        else:
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """Center crop image."""
        h, w = image.shape[-2:]
        ch, cw = crop_size
        
        start_h = (h - ch) // 2
        start_w = (w - cw) // 2
        
        if len(image.shape) == 2:
            return image[start_h:start_h+ch, start_w:start_w+cw]
        elif len(image.shape) == 3:
            return image[:, start_h:start_h+ch, start_w:start_w+cw]
        else:
            return image[:, :, start_h:start_h+ch, start_w:start_w+cw]


class DirManager:
    """Directory management utilities."""
    
    @staticmethod
    def create_dirs(paths: dict):
        """Create multiple directories if they don't exist."""
        for name, path in paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"Directory ready: {name} -> {path}")
    
    @staticmethod
    def save_metrics(metrics: dict, save_path: str):
        """Save metrics to text file."""
        with open(save_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        print(f"Metrics saved to {save_path}")


def get_model_summary(model: torch.nn.Module) -> dict:
    """Get model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable,
    }


def print_model_summary(model: torch.nn.Module):
    """Print model parameter statistics."""
    summary = get_model_summary(model)
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Total Parameters:        {summary['total_parameters']:>12,}")
    print(f"Trainable Parameters:    {summary['trainable_parameters']:>12,}")
    print(f"Non-trainable Parameters:{summary['non_trainable_parameters']:>12,}")
    print("="*50 + "\n")
