"""
Streamlit Web UI for Medical Image Enhancement
Interactive interface for model training visualization and image enhancement with robust error handling.
"""

import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import os
import sys
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.model import UNet, get_model
from src.inference import ImageEnhancer
from src.utils import Visualizer, ImageProcessor, print_model_summary

# Page config
st.set_page_config(
    page_title="X-ray Enhancement Studio",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8em;
        color: #ff7f0e;
        margin-top: 30px;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 10px;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏥 Medical X-ray Enhancement Studio</div>', unsafe_allow_html=True)
st.markdown("*Advanced deep learning for low-data medical image enhancement with robust processing*")

# Initialize session state
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'input_image' not in st.session_state:
    st.session_state.input_image = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Sidebar configuration
with st.sidebar:
    st.markdown("# ⚙️ Settings & Configuration")
    
    # Device selection with validation
    device_option = st.radio(
        "Select Compute Device",
        ["CPU", "GPU (if available)"],
        help="CPU: Slower but always available. GPU: Faster on compatible hardware"
    )
    
    if device_option == "GPU (if available)":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            st.warning("⚠️ CUDA not available. Using CPU instead.")
    else:
        device = "cpu"
    
    device_info = f"📍 Device: **{device.upper()}**"
    if device == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
            device_info += f"\n💾 {device_name}"
            device_info += f"\n📊 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        except:
            pass
    
    st.markdown(device_info)
    
    # Model selection with error handling
    st.markdown("## 📦 Model Selection")
    model_dir = Path("models")
    model_files = []
    model_path = None
    
    try:
        if model_dir.exists():
            model_files = sorted([f for f in model_dir.glob("*.pt") if "checkpoint" not in f.name and f.is_file()])
        
        if model_files:
            selected_model = st.selectbox(
                "Choose a trained model:",
                options=[f.name for f in model_files],
                help="Select a model for inference"
            )
            model_path = model_dir / selected_model
            st.success(f"✅ Model selected: {selected_model}")
        else:
            st.warning("⚠️ No trained models found in ./models directory")
            st.info("💡 Train a model first using train.py")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        logger.error(f"Model loading error: {str(e)}")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Enhancement",
    "📊 Training Metrics",
    "📈 Model Info",
    "📚 Documentation"
])

# TAB 1: Image Enhancement
with tab1:
    st.markdown('<div class="section-header">🎯 Image Enhancement & Inference</div>', unsafe_allow_html=True)
    
    if model_path is None:
        st.error("❌ No model selected. Please ensure a trained model exists in the models directory.")
    else:
        try:
            # Model loading with caching and error handling
            @st.cache_resource
            def load_enhancer(model_path: str, device: str) -> Optional[ImageEnhancer]:
                """Load and cache image enhancer with error handling."""
                try:
                    return ImageEnhancer(str(model_path), device=torch.device(device))
                except Exception as e:
                    logger.error(f"Failed to load enhancer: {str(e)}")
                    raise
            
            # Reload button
            col_reload = st.columns([1, 4, 1])[0]
            with col_reload:
                if st.button("🔄 Reload Model", key="reload_btn"):
                    st.cache_resource.clear()
                    st.rerun()
            
            try:
                enhancer = load_enhancer(str(model_path), device)
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.stop()
            
            # Enhancement interface
            col_input, col_output = st.columns([2, 1])
            
            with col_input:
                st.subheader("📤 Input Image")
                
                input_method = st.radio(
                    "Input Method:",
                    ["Upload Image", "Generate Sample"],
                    horizontal=False
                )
                
                image = None
                
                if input_method == "Upload Image":
                    uploaded_file = st.file_uploader(
                        "Upload X-ray image (PNG, JPG, BMP)",
                        type=['png', 'jpg', 'jpeg', 'bmp'],
                        help="Supported formats: PNG, JPG, JPEG, BMP"
                    )
                    
                    if uploaded_file is not None:
                        try:
                            # Read and validate uploaded image
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                            
                            if image is None:
                                st.error("❌ Failed to read image. Ensure it's a valid image file.")
                            else:
                                image = image.astype(np.float32) / 255.0
                                
                                # Display image info
                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    st.metric("Resolution", f"{image.shape[1]}×{image.shape[0]}")
                                with col_info2:
                                    st.metric("Format", "Grayscale")
                                
                                st.image(np.clip(image, 0, 1), caption="Uploaded Image", use_column_width=True)
                        except Exception as e:
                            st.error(f"❌ Error processing image: {str(e)}")
                            image = None
                else:
                    # Generate sample synthetic X-ray image
                    st.info("📊 Generating synthetic X-ray sample for demonstration...")
                    try:
                        # Create realistic synthetic image
                        image = np.random.rand(256, 256) * 0.5 + 0.2
                        
                        # Add Gaussian blobs to simulate anatomical structures
                        for _ in range(3):
                            y, x = np.random.randint(50, 206, 2)
                            Y, X = np.ogrid[:256, :256]
                            sigma = np.random.randint(30, 80)
                            blob = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
                            intensity = np.random.rand() * 0.4 + 0.1
                            image += blob * intensity
                        
                        image = np.clip(image, 0, 1)
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Resolution", "256×256")
                        with col_info2:
                            st.metric("Format", "Synthetic")
                        
                        st.image(image, caption="Synthetic X-ray Sample", use_column_width=True)
                    except Exception as e:
                        st.error(f"❌ Error generating sample: {str(e)}")
                        image = None
                
                # Enhancement settings
                if image is not None:
                    st.subheader("⚙️ Enhancement Settings")
                    
                    target_size = st.radio(
                        "Processing Resolution:",
                        ["Original", "256×256", "512×512"],
                        horizontal=True
                    )
                    
                    # Enhancement button
                    if st.button("🚀 Enhance Image", key="enhance_btn", use_container_width=True):
                        with st.spinner("🔄 Processing image... This may take a moment."):
                            try:
                                # Determine processing size
                                resize_map = {
                                    "256×256": (256, 256),
                                    "512×512": (512, 512),
                                    "Original": (256, 256)
                                }
                                resize_target = resize_map[target_size]
                                
                                # Validate image before enhancement
                                if not np.isfinite(image).all():
                                    raise ValueError("Image contains invalid values")
                                
                                # Perform enhancement
                                enhanced = enhancer.enhance_image(
                                    image,
                                    resize=resize_target
                                )
                                
                                st.session_state.enhanced_image = enhanced
                                st.session_state.input_image = image
                                st.success("✅ Enhancement completed successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Enhancement failed: {str(e)}")
                                logger.error(f"Enhancement error: {str(e)}")
            
            with col_output:
                st.subheader("📥 Enhanced Output")
                
                if st.session_state.enhanced_image is not None:
                    try:
                        enhanced_clipped = np.clip(st.session_state.enhanced_image.squeeze(), 0, 1)
                        st.image(enhanced_clipped, caption="Enhanced Image", use_column_width=True)
                        
                        # Download button
                        enhanced_uint8 = (enhanced_clipped * 255).astype(np.uint8)
                        _, buffer = cv2.imencode('.png', enhanced_uint8)
                        
                        st.download_button(
                            label="💾 Download Enhanced",
                            data=buffer.tobytes(),
                            file_name="enhanced_xray.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        # Image statistics
                        st.subheader("📊 Image Statistics")
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric(
                                "Mean Value",
                                f"{enhanced_clipped.mean():.4f}",
                                delta=f"{enhanced_clipped.mean() - st.session_state.input_image.mean():.4f}"
                            )
                        with col_stat2:
                            st.metric(
                                "Std Dev",
                                f"{enhanced_clipped.std():.4f}",
                                delta=f"{enhanced_clipped.std() - st.session_state.input_image.std():.4f}"
                            )
                    except Exception as e:
                        st.error(f"❌ Error displaying enhanced image: {str(e)}")
                else:
                    st.info("✨ Enhanced image will appear here after processing")
            
            # Detailed comparison section
            if st.session_state.enhanced_image is not None:
                st.markdown("---")
                st.subheader("🔍 Detailed Comparison & Analysis")
                
                try:
                    col1, col2, col3 = st.columns(3)
                    
                    input_clipped = np.clip(st.session_state.input_image, 0, 1)
                    enhanced_clipped = np.clip(st.session_state.enhanced_image.squeeze(), 0, 1)
                    
                    with col1:
                        st.image(input_clipped, caption="Input (Noisy)", use_column_width=True)
                    
                    with col2:
                        st.image(enhanced_clipped, caption="Enhanced Output", use_column_width=True)
                    
                    with col3:
                        diff = np.abs(enhanced_clipped - input_clipped)
                        st.image(diff, caption="Enhancement Map", use_column_width=True)
                    
                    # Enhancement metrics
                    st.subheader("📈 Enhancement Metrics")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        mse = np.mean((enhanced_clipped - input_clipped) ** 2)
                        st.metric("MSE (Lower is Better)", f"{mse:.6f}")
                    
                    with col_m2:
                        contrast_before = input_clipped.std()
                        contrast_after = enhanced_clipped.std()
                        contrast_ratio = contrast_after / (contrast_before + 1e-8)
                        st.metric("Contrast Enhancement", f"{contrast_ratio:.2f}x", help="Enhancement ratio of output vs input")
                    
                    with col_m3:
                        mean_shift = np.mean(enhanced_clipped) - np.mean(input_clipped)
                        st.metric("Mean Intensity Shift", f"{mean_shift:+.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error displaying comparison: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error in enhancement section: {str(e)}")
            logger.error(f"Enhancement section error: {str(e)}")

# TAB 2: Training Metrics
with tab2:
    st.markdown('<div class="section-header">📊 Training Metrics & Progress</div>', unsafe_allow_html=True)
    
    logs_dir = Path("logs")
    
    try:
        if not logs_dir.exists():
            st.warning("⚠️ No logs directory found. Train a model to generate metrics.")
        else:
            history_files = sorted([f for f in logs_dir.glob("history_*.json") if f.is_file()])
            
            if not history_files:
                st.info("💡 No training history files found. Start training to generate metrics.")
            else:
                selected_history = st.selectbox(
                    "📂 Select training session:",
                    options=[f.name for f in history_files],
                    format_func=lambda x: x.replace("history_", "").replace(".json", "")
                )
                
                try:
                    with open(logs_dir / selected_history, 'r') as f:
                        history = json.load(f)
                    
                    # Validate history data
                    if 'train_loss' not in history or 'val_loss' not in history:
                        st.error("❌ Invalid history format: missing required fields")
                    else:
                        # Training progress visualization
                        st.subheader("📈 Loss Progression")
                        
                        col_chart, col_info = st.columns([2, 1])
                        
                        with col_chart:
                            train_losses = history.get('train_loss', [])
                            val_losses = history.get('val_loss', [])
                            
                            if train_losses and val_losses:
                                import pandas as pd
                                loss_df = pd.DataFrame({
                                    'Epoch': range(len(train_losses)),
                                    'Train Loss': train_losses,
                                    'Val Loss': val_losses
                                })
                                st.line_chart(loss_df.set_index('Epoch'))
                            else:
                                st.warning("⚠️ No loss data available")
                        
                        with col_info:
                            st.subheader("📊 Summary")
                            if train_losses:
                                st.metric("Starting Loss", f"{train_losses[0]:.6f}")
                                st.metric("Final Train Loss", f"{train_losses[-1]:.6f}")
                            if val_losses:
                                st.metric("Final Val Loss", f"{val_losses[-1]:.6f}")
                            st.metric("Total Epochs", len(train_losses))
                        
                        # Image quality metrics
                        if 'val_metrics' in history and history['val_metrics']:
                            st.subheader("🎯 Image Quality Metrics")
                            
                            metrics_count = len(history['val_metrics'])
                            last_metrics = history['val_metrics'][-1] if history['val_metrics'] else {}
                            
                            col_m1, col_m2, col_m3 = st.columns(3)
                            
                            with col_m1:
                                if 'psnr' in last_metrics:
                                    psnr_vals = [m.get('psnr', 0) for m in history['val_metrics']]
                                    st.metric(
                                        "PSNR (dB)",
                                        f"{last_metrics['psnr']:.4f}",
                                        delta=f"{last_metrics['psnr'] - psnr_vals[0]:.4f}" if len(psnr_vals) > 1 else None,
                                        help="Peak Signal-to-Noise Ratio - higher is better"
                                    )
                            
                            with col_m2:
                                if 'ssim' in last_metrics:
                                    ssim_vals = [m.get('ssim', 0) for m in history['val_metrics']]
                                    st.metric(
                                        "SSIM",
                                        f"{last_metrics['ssim']:.4f}",
                                        delta=f"{last_metrics['ssim'] - ssim_vals[0]:.4f}" if len(ssim_vals) > 1 else None,
                                        help="Structural Similarity Index - higher is better (max=1.0)"
                                    )
                            
                            with col_m3:
                                if 'mse' in last_metrics:
                                    mse_vals = [m.get('mse', 0) for m in history['val_metrics']]
                                    st.metric(
                                        "MSE",
                                        f"{last_metrics['mse']:.6f}",
                                        delta=f"{last_metrics['mse'] - mse_vals[0]:.6f}" if len(mse_vals) > 1 else None,
                                        help="Mean Squared Error - lower is better"
                                    )
                        
                        # Learning rate info
                        if 'learning_rate' in history:
                            st.subheader("🔧 Training Configuration")
                            st.info(f"📝 Training epochs: {len(history['train_loss'])}")
                
                except json.JSONDecodeError:
                    st.error("❌ Error reading history file: Invalid JSON format")
                except Exception as e:
                    st.error(f"❌ Error loading history: {str(e)}")
                    logger.error(f"History loading error: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error in Training Metrics tab: {str(e)}")
        logger.error(f"Training metrics tab error: {str(e)}")

# TAB 3: Model Information
with tab3:
    st.markdown('<div class="section-header">📈 Model Architecture & Information</div>', unsafe_allow_html=True)
    
    try:
        if model_path is None:
            st.warning("⚠️ No model loaded. Please select a model to view details.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📋 Model Statistics")
                
                try:
                    device_obj = torch.device(device)
                    model = UNet(in_channels=1, out_channels=1, depth=3, dropout=0.2).to(device_obj)
                    
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    non_trainable = total_params - trainable_params
                    
                    st.metric("Total Parameters", f"{total_params:,}")
                    st.metric("Trainable Params", f"{trainable_params:,}")
                    st.metric("Non-trainable Params", f"{non_trainable:,}")
                    
                    # Model size
                    model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
                    st.metric("Model Size (approx)", f"{model_size_mb:.2f} MB")
                except Exception as e:
                    st.error(f"❌ Error calculating model stats: {str(e)}")
            
            with col2:
                st.subheader("🏗️ Architecture Details")
                st.code("""
Architecture: UNet
─────────────────────
Encoder:
  • Level 1: 1   → 32  channels
  • Level 2: 32  → 64  channels
  • Level 3: 64  → 128 channels

Bottleneck: 128 → 256 channels

Decoder:
  • Level 3: 256 → 128 channels
  • Level 2: 128 → 64  channels
  • Level 1: 64  → 32  channels

Output: 32 → 1 channel (Grayscale)

Features:
  ✓ Skip connections
  ✓ Batch normalization
  ✓ Dropout regularization
  ✓ ReLU activations
                """, language="text")
            
            # Configuration section
            st.subheader("⚙️ Training Configuration")
            
            config_path = Path("configs/config.yaml")
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                    
                    with col_cfg1:
                        st.metric(
                            "Epochs",
                            config.get('training', {}).get('epochs', 'N/A')
                        )
                        st.metric(
                            "Learning Rate",
                            f"{config.get('training', {}).get('learning_rate', 0):.6f}"
                        )
                    
                    with col_cfg2:
                        st.metric(
                            "Optimizer",
                            config.get('training', {}).get('optimizer', 'N/A')
                        )
                        st.metric(
                            "Loss Function",
                            config.get('training', {}).get('loss_function', 'N/A')
                        )
                    
                    with col_cfg3:
                        st.metric(
                            "Batch Size",
                            config.get('data', {}).get('batch_size', 'N/A')
                        )
                        st.metric(
                            "Early Stopping Patience",
                            config.get('training', {}).get('early_stopping_patience', 'N/A')
                        )
                    
                    # Model config details
                    st.info(
                        f"""**Model Configuration:**
- In Channels: {config.get('model', {}).get('in_channels', 1)}
- Out Channels: {config.get('model', {}).get('out_channels', 1)}
- Depth: {config.get('model', {}).get('depth', 3)}
- Dropout: {config.get('model', {}).get('dropout', 0.2)}
                        """
                    )
                except Exception as e:
                    st.warning(f"Could not load configuration: {str(e)}")
            else:
                st.warning("Configuration file not found")
    
    except Exception as e:
        st.error(f"❌ Error in Model Information tab: {str(e)}")
        logger.error(f"Model info tab error: {str(e)}")
    
    # System info
    st.markdown("---")
    st.subheader("💻 System & Runtime Information")
    
    col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)
    
    with col_sys1:
        st.metric("PyTorch", torch.__version__.split('+')[0])
    
    with col_sys2:
        cuda_status = "Available" if torch.cuda.is_available() else "Not Available"
        st.metric("CUDA", cuda_status)
    
    with col_sys3:
        st.metric("Device", device.upper())
    
    with col_sys4:
        if device == "cuda":
            try:
                cuda_name = torch.cuda.get_device_name(0)
                st.metric("GPU", cuda_name[:25])
            except:
                st.metric("GPU", "N/A")

# TAB 4: Documentation
with tab4:
    st.markdown('<div class="section-header">📚 Documentation & Usage Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🚀 Quick Start Guide
    
    ### 1. **Training Your Model**
    
    Run the proof of concept training script:
    ```bash
    python src/poc.py
    ```
    
    This will:
    - Generate synthetic X-ray dataset
    - Train a UNet model for 100 epochs
    - Save checkpoints every 10 epochs
    - Generate training metrics and visualizations
    
    ### 2. **Enhancing Images**
    
    #### Using the Web UI (You are here!)
    - Go to the **Enhancement** tab
    - Upload an X-ray image or use a sample
    - Click "Enhance Image"
    - Download the enhanced result
    
    #### Using Command Line
    ```bash
    python src/inference.py --input image.jpg --output enhanced.jpg --visualize
    ```
    
    #### Using Python API
    ```python
    from src.inference import ImageEnhancer
    
    enhancer = ImageEnhancer('models/model_final.pt')
    enhanced = enhancer.enhance_from_file('xray.jpg', 'output.jpg')
    ```
    
    ### 3. **Project Structure**
    
    ```
    ├── src/
    │   ├── model.py          # UNet and variant architectures
    │   ├── poc.py            # Training proof of concept
    │   ├── train.py          # Advanced trainer with callbacks
    │   ├── inference.py      # Inference engine
    │   ├── evaluation.py     # Metrics and loss functions
    │   └── utils.py          # Helper utilities
    ├── configs/
    │   └── config.yaml       # Training configuration
    ├── models/               # Trained model checkpoints
    ├── outputs/              # Generated visualizations
    ├── logs/                 # Training history
    └── app.py               # This Streamlit web UI
    ```
    
    ## 🔧 Configuration
    
    Edit `configs/config.yaml` to customize:
    - Model architecture (UNet, ResUNet)
    - Training parameters (epochs, learning rate)
    - Loss functions (MSE, L1, SSIM, Perceptual)
    - Data augmentation settings
    - Optimizers and schedulers
    
    ## 🎯 Model Architectures Available
    
    1. **UNet** - Classic U-shaped architecture with skip connections
    2. **ResUNet** - UNet with residual blocks for better gradient flow
    3. **XrayEnhancementNet** - Specialized multi-scale network
    
    ## 📊 Loss Functions
    
    - **MSE** - Mean Squared Error (default)
    - **L1** - Mean Absolute Error
    - **SSIM** - Structural Similarity Index
    - **Perceptual** - Edge-based perceptual loss
    
    ## 💡 Tips for Best Results
    
    1. **Data Quality**: Ensure input images are properly normalized (0-1 range)
    2. **Resolution**: For best results, use images of consistent size (256x256 recommended)
    3. **Batch Size**: Adjust based on available GPU memory
    4. **Learning Rate**: Use the scheduler for adaptive learning rates
    5. **Early Stopping**: Training stops if validation loss doesn't improve for 15 epochs
    
    ## 🔍 Monitoring Training
    
    During training, check these files:
    - `models/checkpoint_epoch_*.pt` - Model checkpoints
    - `logs/history_*.json` - Training metrics
    - `outputs/training_plot_*.png` - Loss visualization
    
    ## 📧 Support & Troubleshooting
    
    - **Out of Memory**: Reduce batch size in config.yaml
    - **Slow Training**: Switch to GPU if available
    - **Poor Results**: Check input image normalization and quality
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>🏥 Medical Image Enhancement | Low-Data Deep Learning | X-ray Processing</p>
    <p>Built with Streamlit • PyTorch • Python</p>
</div>
""", unsafe_allow_html=True)
