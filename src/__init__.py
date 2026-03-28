"""
Medical X-ray Image Enhancement Package
Low-data deep learning solution for X-ray image quality improvement.
"""

import logging
import logging.handlers
from pathlib import Path

__version__ = "0.2.0"
__author__ = "X-ray Enhancement Team"

# Configure logging for the package
def _setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Setup comprehensive logging configuration for the package.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
    """
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / 'xray_enhancement.log',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized: {__name__} v{__version__}")

# Initialize logging on package import
_setup_logging()

# Import module components
from .model import UNet, ResUNet, XrayEnhancementNet, get_model
from .train import Trainer, TrainingConfig
from .inference import ImageEnhancer
from .evaluation import EarlyStopping, ModelEvaluator, PerceptualLoss, SSIMLoss
from .utils import (
    ImageMetrics, Visualizer, ImageProcessor, DirManager,
    get_model_summary, print_model_summary
)

__all__ = [
    'UNet', 'ResUNet', 'XrayEnhancementNet', 'get_model',
    'Trainer', 'TrainingConfig',
    'ImageEnhancer',
    'EarlyStopping', 'ModelEvaluator', 'PerceptualLoss', 'SSIMLoss',
    'ImageMetrics', 'Visualizer', 'ImageProcessor', 'DirManager',
    'get_model_summary', 'print_model_summary'
]
