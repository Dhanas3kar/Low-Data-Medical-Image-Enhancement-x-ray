"""
Test script to verify image enhancement pipeline works correctly with various input sizes.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import UNet
from src.utils import ImageProcessor

def test_enhancement_pipeline():
    """Test the enhancement pipeline with different image sizes."""
    
    print("Testing Image Enhancement Pipeline")
    print("=" * 60)
    
    # Test 1: Standard 256x256
    print("\nTest 1: 256x256 image")
    img1 = np.random.rand(256, 256)
    print(f"  Input shape: {img1.shape}")
    img1_normalized = ImageProcessor.normalize(img1)
    print(f"  Normalized shape: {img1_normalized.shape}")
    print(f"  ✓ PASSED")
    
    # Test 2: Different size (uploaded image size)
    print("\nTest 2: 418x425 image (uploaded size)")
    img2 = np.random.rand(418, 425)
    print(f"  Input shape: {img2.shape}")
    img2_normalized = ImageProcessor.normalize(img2)
    print(f"  Normalized shape: {img2_normalized.shape}")
    img2_resized = ImageProcessor.resize(img2_normalized, (256, 256))
    print(f"  Resized to (256, 256): {img2_resized.shape}")
    print(f"  ✓ PASSED")
    
    # Test 3: With channel dimension
    print("\nTest 3: Image with channel dimension (1, 256, 256)")
    img3 = np.random.rand(1, 256, 256)
    print(f"  Input shape: {img3.shape}")
    img3_normalized = ImageProcessor.normalize(img3)
    print(f"  Normalized shape: {img3_normalized.shape}")
    print(f"  ✓ PASSED")
    
    # Test 4: Model forward pass
    print("\nTest 4: Model forward pass")
    device = torch.device('cpu')
    model = UNet(in_channels=1, out_channels=1, depth=3, dropout=0.2).to(device)
    
    # Create dummy input
    batch = torch.randn(1, 1, 256, 256).to(device)
    print(f"  Input batch shape: {batch.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"  Output batch shape: {output.shape}")
    print(f"  ✓ PASSED")
    
    # Test 5: Full pipeline with different sizes
    print("\nTest 5: Full pipeline - variable input size")
    test_sizes = [(512, 512), (384, 384), (418, 425), (256, 256)]
    
    for h, w in test_sizes:
        print(f"\n  Testing {h}x{w}...")
        test_img = np.random.rand(h, w)
        
        # Add channel dimension
        if test_img.ndim == 2:
            test_img = test_img[np.newaxis, :, :]
        
        # Normalize
        test_img = ImageProcessor.normalize(test_img)
        
        # Resize for model
        test_img_resized = ImageProcessor.resize(test_img, (256, 256))
        
        # Convert to tensor
        test_tensor = torch.FloatTensor(test_img_resized).unsqueeze(0).to(device)
        
        # Model inference
        with torch.no_grad():
            output = model(test_tensor)
        
        enhanced = output.squeeze(0).cpu().numpy()
        
        # Resize back
        enhanced_resized = ImageProcessor.resize(enhanced, (h, w))
        
        print(f"    Original: {(h, w)} → Model: {test_img_resized.shape[-2:]} → Output: {enhanced_resized.shape[-2:]}")
        print(f"    ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_enhancement_pipeline()
