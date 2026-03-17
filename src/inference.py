"""
Inference script for image enhancement.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import yaml
import cv2

from .model import UNet
from .utils import ImageProcessor, Visualizer, print_model_summary


class ImageEnhancer:
    """Image enhancement inference class."""
    
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml',
                 device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = UNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels'],
            depth=self.config['model']['depth'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Load weights
        if Path(model_path).exists():
            weights = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(weights)
            print(f"✓ Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model.eval()
        print_model_summary(self.model)
    
    def enhance_image(self, image: np.ndarray, resize: Optional[Tuple[int, int]] = None,
                     return_all: bool = False) -> np.ndarray:
        """
        Enhance single image with automatic size handling.
        
        Args:
            image: Input image (H, W) or (C, H, W)
            resize: Optional resize target (H, W). If None, uses 256x256
            return_all: If True, return (input, enhanced) tuple
        
        Returns:
            Enhanced image or tuple of (input, enhanced)
        """
        # Prepare input
        if image.ndim == 2:
            image = image[np.newaxis, :, :]  # Add channel dimension
        
        # Store original size
        orig_shape = image.shape
        
        # Normalize to [0, 1]
        image = ImageProcessor.normalize(image)
        
        # Determine resize target (default to 256x256)
        if resize is None:
            resize = (256, 256)
        
        # Resize for model input
        image_resized = ImageProcessor.resize(image, resize)
        
        # Add batch dimension for model
        image_tensor = torch.FloatTensor(image_resized).unsqueeze(0).to(self.device)
        
        # Model inference
        with torch.no_grad():
            enhanced_tensor = self.model(image_tensor)
        
        # Remove batch dimension and clip
        enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)
        
        # Resize back to original size
        if enhanced.shape[-2:] != orig_shape[-2:]:
            enhanced = ImageProcessor.resize(enhanced, (orig_shape[-2], orig_shape[-1]))
        
        if return_all:
            return image, enhanced
        return enhanced
    
    def enhance_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Enhance batch of images.
        
        Args:
            images: Batch of images (B, C, H, W) or (B, H, W)
        
        Returns:
            Enhanced batch
        """
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]  # Add channel dimension
        
        # Normalize
        images = np.array([ImageProcessor.normalize(img) for img in images])
        
        # Convert to tensor
        images_tensor = torch.FloatTensor(images).to(self.device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor = self.model(images_tensor)
        
        enhanced = enhanced_tensor.cpu().numpy()
        return np.clip(enhanced, 0, 1)
    
    def enhance_from_file(self, image_path: str, save_path: Optional[str] = None,
                         visualize: bool = True) -> np.ndarray:
        """
        Load image from file, enhance it, and optionally save.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save enhanced image
            visualize: Whether to create visualization
        
        Returns:
            Enhanced image
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Enhance
        print(f"Enhancing image: {image_path}")
        enhanced = self.enhance_image(img)
        
        # Save enhanced image
        if save_path:
            save_img = (enhanced.squeeze() * 255).astype(np.uint8)
            cv2.imwrite(save_path, save_img)
            print(f"✓ Enhanced image saved to: {save_path}")
        
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
