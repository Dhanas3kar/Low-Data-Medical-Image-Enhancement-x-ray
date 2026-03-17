# Low-Data Medical Image Enhancement - X-ray

A lightweight deep learning solution for enhancing X-ray images with limited training data. This project demonstrates a practical approach to medical image quality improvement using efficient neural network architectures.

## Features

- **Lightweight UNet Architecture**: Optimized for low-data scenarios
- **Configurable Pipeline**: YAML-based configuration for easy experimentation
- **Synthetic Dataset Support**: POC includes synthetic X-ray generation
- **Training & Inference**: Complete training loop and enhancement inference
- **Extensible Design**: Easy to integrate real medical data

## Project Structure

```
├── src/
│   ├── model.py          # UNet architecture definition
│   └── poc.py            # Proof of concept training script
├── configs/
│   └── config.yaml       # Main configuration file
├── data/                 # Dataset directory (to be populated)
├── models/               # Trained model checkpoints
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Configuration

The main configuration is defined in [configs/config.yaml](configs/config.yaml). Key settings:

- **Data**: Image size, batch size, train/val/test splits
- **Model**: Architecture, depth, dropout, channels
- **Training**: Epochs, learning rate, optimizer, loss function
- **Augmentation**: Image transformation parameters
- **Paths**: Data, models, outputs directories

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Proof of Concept

```bash
python src/poc.py
```

This will:
- Generate synthetic X-ray image pairs
- Train the UNet model for 50 epochs
- Visualize training progress and enhancement results
- Save results to `poc_results.png`

### 3. Use Pre-trained Model (When Available)

```python
from src.model import UNet
import torch

model = UNet()
model.load_state_dict(torch.load('models/checkpoint.pth'))
model.eval()

# Enhance image
enhanced = model(low_quality_image)
```

## Model Details

**Architecture**: UNet with 3 encoding layers
- Parameters: ~2.5M (lightweight for medical imaging)
- Input: 1-channel (grayscale) X-ray images
- Output: 1-channel enhanced X-ray images
- Supports batch processing and GPU acceleration

## Future Enhancements

- [ ] Real medical X-ray dataset integration
- [ ] Advanced loss functions (SSIM, Perceptual)
- [ ] Data augmentation strategies for low-data scenarios
- [ ] Model compression and optimization
- [ ] Inference API and web interface
- [ ] Multi-resolution enhancement
- [ ] Transfer learning from natural images

## Hardware Requirements

- GPU: Recommended (NVIDIA, AMD, or Apple Silicon)
- CPU: Fallback supported (slower training)
- Memory: 4GB+ (8GB+ recommended)

## References

- UNet: Ronneberger et al., 2015
- Medical Image Enhancement: Common architectures and techniques
- Low-Data Learning: Data augmentation and transfer learning strategies

## License

MIT License - See LICENSE file for details

## Contact

For questions or contributions, please open an issue or pull request.
