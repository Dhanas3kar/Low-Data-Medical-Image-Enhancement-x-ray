"""
Inference script for image enhancement with robust error handling.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml
import cv2
import logging

from .model import UNet, get_model
from .utils import ImageProcessor, Visualizer, print_model_summary

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Image enhancement inference class with robust error handling."""
    
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml',
                 device: Optional[torch.device] = None, architecture: str = 'UNet'):
        """
        Initialize ImageEnhancer with model loading and configuration.
        
        Args:
            model_path: Path to model weights
            config_path: Path to config YAML
            device: Torch device (cpu/cuda)
            architecture: Model architecture name
            
        Raises:
            FileNotFoundError: If model or config not found
            ValueError: If config is invalid
        """
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load and validate config
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Config loaded from: {config_path}")
        except Exception as e:
            raise ValueError(f"Failed to load config: {str(e)}")
        
        # Validate config structure
        required_keys = {'model', 'inference'}
        if not all(key in self.config for key in required_keys):
            raise ValueError(f"Config missing required keys: {required_keys}")
        
        # Initialize model with error handling
        try:
            self.model = get_model(
                architecture=architecture,
                in_channels=self.config['model'].get('in_channels', 1),
                out_channels=self.config['model'].get('out_channels', 1),
                depth=self.config['model'].get('depth', 3),
                dropout=self.config['model'].get('dropout', 0.2)
            ).to(self.device)
            logger.info(f"Model initialized: {architecture}")
        except Exception as e:
            raise ValueError(f"Failed to initialize model: {str(e)}")
        
        # Load weights with validation
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        try:
            weights = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(weights)
            logger.info(f"Model weights loaded: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        
        self.model.eval()
        print_model_summary(self.model)
    
    def enhance_image(self, image: np.ndarray, resize: Optional[Tuple[int, int]] = None,
                     return_all: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Enhance single image with automatic size handling and validation.
        
        Args:
            image: Input image (H, W) or (C, H, W)
            resize: Optional resize target (H, W). If None, uses 256x256
            return_all: If True, return (input, enhanced) tuple
        
        Returns:
            Enhanced image or tuple of (input, enhanced)
            
        Raises:
            TypeError: If image is not ndarray
            ValueError: If image has invalid shape or is empty
        """
        # Input validation
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image)}")
        
        if image.size == 0:
            raise ValueError("Image is empty")
        
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")
        
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or infinite values")
        
        # Prepare input
        if image.ndim == 2:
            image = image[np.newaxis, :, :]  # Add channel dimension
        
        # Store original size
        orig_shape = image.shape
        
        # Normalize to [0, 1]
        image_normalized = ImageProcessor.normalize(image)
        
        # Determine resize target (default to 256x256)
        if resize is None:
            resize = (256, 256)
        elif not isinstance(resize, tuple) or len(resize) != 2:
            raise ValueError(f"resize must be tuple of 2 ints, got {resize}")
        
        # Resize for model input
        try:
            image_resized = ImageProcessor.resize(image_normalized, resize)
        except Exception as e:
            raise RuntimeError(f"Failed to resize image: {str(e)}")
        
        # Add batch dimension for model
        try:
            image_tensor = torch.FloatTensor(image_resized).unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to convert image to tensor: {str(e)}")
        
        # Model inference
        try:
            with torch.no_grad():
                enhanced_tensor = self.model(image_tensor)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        # Post-processing
        try:
            enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
            enhanced = np.clip(enhanced, 0, 1)
            
            # Resize back to original size if needed
            if enhanced.shape[-2:] != orig_shape[-2:]:
                enhanced = ImageProcessor.resize(enhanced, (orig_shape[-2], orig_shape[-1]))
        except Exception as e:
            raise RuntimeError(f"Post-processing failed: {str(e)}")
        
        logger.info(f"Image enhanced successfully. Shape: {enhanced.shape}")
        
        if return_all:
            return image_normalized, enhanced
        return enhanced
    
    def enhance_batch(self, images: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Enhance batch of images with memory-efficient processing.
        
        Args:
            images: Batch of images (B, C, H, W) or (B, H, W)
            batch_size: Processing batch size (for memory efficiency)
        
        Returns:
            Enhanced batch
            
        Raises:
            TypeError: If images is not ndarray
            ValueError: If images has invalid shape
        """
        if not isinstance(images, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(images)}")
        
        if images.size == 0:
            raise ValueError("Images array is empty")
        
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]  # Add channel dimension
        elif images.ndim != 4:
            raise ValueError(f"Images must be 3D or 4D, got {images.ndim}D")
        
        try:
            # Normalize
            normalized = np.array([ImageProcessor.normalize(img) for img in images])
            
            # Use config batch size if not specified
            if batch_size is None:
                batch_size = self.config['inference'].get('batch_size', 4)
            
            # Process in batches
            enhanced_list = []
            for i in range(0, len(normalized), batch_size):
                batch = normalized[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                with torch.no_grad():
                    enhanced_tensor = self.model(batch_tensor)
                
                enhanced_list.append(enhanced_tensor.cpu().numpy())
            
            enhanced = np.vstack(enhanced_list)
            logger.info(f"Batch enhanced: {enhanced.shape}")
            return np.clip(enhanced, 0, 1)
        except Exception as e:
            raise RuntimeError(f"Batch enhancement failed: {str(e)}")
    
    def enhance_from_file(self, image_path: str, save_path: Optional[str] = None,
                         visualize: bool = True) -> np.ndarray:
        """
        Load image from file, enhance it, and optionally save with error handling.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save enhanced image
            visualize: Whether to create visualization
        
        Returns:
            Enhanced image
            
        Raises:
            FileNotFoundError: If image not found
            ValueError: If image loading fails
        """
        # Validate input path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image with error handling
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image or unsupported format")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
        
        # Convert to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Enhance
        try:
            logger.info(f"Enhancing image: {image_path}")
            enhanced = self.enhance_image(img)
        except Exception as e:
            raise RuntimeError(f"Enhancement failed: {str(e)}")
        
        # Save enhanced image
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_img = (enhanced.squeeze() * 255).astype(np.uint8)
                cv2.imwrite(str(save_path), save_img)
                logger.info(f"Enhanced image saved: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save enhanced image: {str(e)}")
                raise RuntimeError(f"Failed to save image: {str(e)}")
        
        # Visualization
        if visualize:
            input_img, output_img = self.enhance_image(img, return_all=True)
            Visualizer.plot_enhancement_results(
                input_img,
                output_img,
                save_path=str(Path(save_path).parent / 'enhancement_comparison.png') if save_path else None
            )
        
        return enhanced
    
    def enhance_dataset(self, input_dir: str, output_dir: str, visualize: bool = False):
        """
        Enhance all images in directory recursively.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save enhanced images
            visualize: Whether to create visualizations
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"\nEnhancing {len(image_files)} images...")
        
        for i, img_path in enumerate(image_files, 1):
            try:
                # Maintain directory structure
                rel_path = img_path.relative_to(input_path)
                save_dir = output_path / rel_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / f"enhanced_{img_path.name}"
                
                print(f"[{i}/{len(image_files)}] ", end="")
                self.enhance_from_file(str(img_path), str(save_path), visualize=visualize)
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"\n✓ All images enhanced and saved to: {output_dir}")


def main():
    """Command-line interface for image enhancement."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhance medical X-ray images using trained model'
    )
    parser.add_argument('--model', type=str, default='models/model_final.pt',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--input', type=str, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output path for enhanced image(s)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create comparison visualization')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize enhancer
    print("Loading model...")
    enhancer = ImageEnhancer(args.model, args.config, device)
    
    # Process input
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            output_path = args.output or f"enhanced_{input_path.name}"
            enhancer.enhance_from_file(args.input, output_path, visualize=args.visualize)
        
        elif input_path.is_dir():
            # Directory
            output_dir = args.output or 'enhanced_output'
            enhancer.enhance_dataset(args.input, output_dir, visualize=args.visualize)
        
        else:
            print(f"Input path not found: {args.input}")
    
    else:
        print("\n" + "="*60)
        print("IMAGE ENHANCEMENT INFERENCE")
        print("="*60)
        print(f"Model loaded: {args.model}")
        print(f"Device: {device}")
        print("\nUsage:")
        print("  Single image:  python src/inference.py --input img.jpg --output enhanced.jpg")
        print("  Directory:     python src/inference.py --input ./data --output ./enhanced")
        print("  Add --visualize to create comparison plots")


if __name__ == "__main__":
    main()
