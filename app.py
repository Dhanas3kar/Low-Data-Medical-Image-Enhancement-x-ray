"""
Streamlit Web UI for Medical Image Enhancement
Interactive interface for model training visualization and image enhancement.
"""

import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.model import UNet
from src.inference import ImageEnhancer
from src.utils import Visualizer, ImageProcessor, print_model_summary


# Page config
st.set_page_config(
    page_title="X-ray Enhancement Studio",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 1.8em;
        color: #ff7f0e;
        margin-top: 30px;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 10px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏥 Medical X-ray Enhancement Studio</div>', unsafe_allow_html=True)
st.markdown("*Advanced deep learning for low-data medical image enhancement*")

# Sidebar
with st.sidebar:
    st.markdown("# ⚙️ Settings")
    
    device = st.radio("Select Device", ["CPU", "GPU (if available)"])
    if device == "GPU (if available)":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    
    st.info(f"📍 Using device: **{device.upper()}**")
    
    # Model selection
    st.markdown("## Model Selection")
    model_dir = Path("models")
    model_files = []
    if model_dir.exists():
        model_files = sorted([f for f in model_dir.glob("*.pt") if "checkpoint" not in f.name])
    
    if model_files:
        selected_model = st.selectbox(
            "Choose Model",
            options=[f.name for f in model_files],
            help="Select a trained model for inference"
        )
        model_path = model_dir / selected_model
    else:
        st.warning("⚠️ No trained models found. Train a model first!")
        model_path = None

# Create tabs
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
        st.error("Please train a model first using the training scripts!")
    else:
        try:
            # Load model
            @st.cache_resource
            def load_enhancer(model_path, device):
                return ImageEnhancer(str(model_path), device=torch.device(device))
            
            # Clear cache if needed (for debugging)
            if st.sidebar.button("🔄 Reload Model"):
                st.cache_resource.clear()
                st.rerun()
            
            enhancer = load_enhancer(model_path, device)
            
            # Enhancement options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📤 Input Image")
                
                input_method = st.radio("Choose input method:", ["Upload Image", "Use Sample"])
                
                if input_method == "Upload Image":
                    uploaded_file = st.file_uploader(
                        "Upload X-ray image (PNG, JPG, BMP)",
                        type=['png', 'jpg', 'jpeg', 'bmp']
                    )
                    
                    if uploaded_file:
                        # Read uploaded image
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                        image = image.astype(np.float32) / 255.0
                        st.image(np.clip(image, 0, 1), caption="Uploaded Image", use_column_width=True)
                else:
                    # Generate sample synthetic image
                    st.info("Generating synthetic X-ray sample...")
                    image = np.random.rand(256, 256) * 0.7
                    # Add some structure
                    y, x = np.random.randint(50, 200, 2)
                    Y, X = np.ogrid[:256, :256]
                    mask = (X - x)**2 + (Y - y)**2 <= 60**2
                    image[mask] += 0.3
                    image = np.clip(image, 0, 1)
                    st.image(image, caption="Sample Synthetic X-ray", use_column_width=True)
                
                # Enhancement parameters
                st.subheader("⚙️ Enhancement Settings")
                target_size = st.selectbox(
                    "Target size (resize before enhancement):",
                    ["Original", "256x256", "512x512"]
                )
                
                if st.button("🚀 Enhance Image", key="enhance_btn"):
                    with st.spinner("Enhancing image..."):
                        try:
                            # Determine processing size
                            if target_size == "256x256":
                                resize_target = (256, 256)
                            elif target_size == "512x512":
                                resize_target = (512, 512)
                            else:
                                # "Original" - will be resized to 256x256 for processing, then back
                                resize_target = (256, 256)
                            
                            # Enhance with automatic size handling
                            enhanced = enhancer.enhance_image(
                                image, 
                                resize=resize_target
                            )
                            st.session_state.enhanced_image = enhanced
                            st.session_state.input_image = image
                            st.success("✅ Enhancement complete!")
                        except Exception as e:
                            st.error(f"❌ Enhancement failed: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
            
            with col2:
                st.subheader("📥 Output Image")
                
                if 'enhanced_image' in st.session_state:
                    enhanced = st.session_state.enhanced_image
                    st.image(np.clip(enhanced.squeeze(), 0, 1), caption="Enhanced Image", use_column_width=True)
                    
                    # Download button
                    enhanced_uint8 = (np.clip(enhanced.squeeze(), 0, 1) * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.png', enhanced_uint8)
                    st.download_button(
                        label="💾 Download Enhanced Image",
                        data=buffer.tobytes(),
                        file_name="enhanced_image.png",
                        mime="image/png"
                    )
                else:
                    st.info("Enhanced image will appear here")
            
            # Comparison view
            if 'enhanced_image' in st.session_state:
                st.subheader("🔍 Detailed Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(np.clip(st.session_state.input_image, 0, 1), caption="Input (Noisy)")
                
                with col2:
                    st.image(np.clip(st.session_state.enhanced_image.squeeze(), 0, 1), caption="Enhanced Output")
                
                with col3:
                    diff = np.abs(st.session_state.enhanced_image.squeeze() - st.session_state.input_image)
                    st.image(diff, caption="Enhancement Map (Difference)")
                
                # Statistics
                st.subheader("📊 Enhancement Statistics")
                enhancement_col1, enhancement_col2 = st.columns(2)
                
                with enhancement_col1:
                    mse = np.mean((st.session_state.enhanced_image.squeeze() - st.session_state.input_image) ** 2)
                    st.metric("Mean Squared Error", f"{mse:.6f}", delta="-" if mse < 0.1 else None)
                
                with enhancement_col2:
                    contrast_before = st.session_state.input_image.std()
                    contrast_after = st.session_state.enhanced_image.squeeze().std()
                    st.metric("Contrast Enhanced", f"{contrast_after/contrast_before:.2f}x")
        
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

# TAB 2: Training Metrics
with tab2:
    st.markdown('<div class="section-header">📊 Training Metrics & Progress</div>', unsafe_allow_html=True)
    
    logs_dir = Path("logs")
    if logs_dir.exists():
        history_files = sorted(logs_dir.glob("history_*.json"))
        
        if history_files:
            selected_history = st.selectbox(
                "Select training history:",
                options=[f.name for f in history_files]
            )
            
            with open(logs_dir / selected_history, 'r') as f:
                history = json.load(f)
            
            # Training progress
            st.subheader("📈 Loss Progression")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart({
                    'Train Loss': history.get('train_loss', []),
                    'Val Loss': history.get('val_loss', [])
                })
            
            with col2:
                st.info("""
                **Key Metrics:**
                - 📉 Starting Train Loss: {:.6f}
                - 📉 Final Train Loss: {:.6f}
                - 📉 Final Val Loss: {:.6f}
                - ✅ Total Epochs: {}
                """.format(
                    history['train_loss'][0] if history['train_loss'] else 0,
                    history['train_loss'][-1] if history['train_loss'] else 0,
                    history['val_loss'][-1] if history['val_loss'] else 0,
                    len(history['train_loss'])
                ))
            
            # Additional metrics
            if 'val_metrics' in history and history['val_metrics']:
                st.subheader("🎯 Image Quality Metrics")
                
                last_metrics = history['val_metrics'][-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'psnr' in last_metrics:
                        st.metric("PSNR (Peak Signal-to-Noise Ratio)", 
                                f"{last_metrics['psnr']:.4f}")
                
                with col2:
                    if 'ssim' in last_metrics:
                        st.metric("SSIM (Structural Similarity)", 
                                f"{last_metrics['ssim']:.4f}")
                
                with col3:
                    if 'mse' in last_metrics:
                        st.metric("MSE (Mean Squared Error)", 
                                f"{last_metrics['mse']:.6f}")
        else:
            st.info("ℹ️ No training history found. Start training to see metrics!")
    else:
        st.warning("⚠️ No logs directory found")

# TAB 3: Model Information
with tab3:
    st.markdown('<div class="section-header">📈 Model Architecture & Info</div>', unsafe_allow_html=True)
    
    if model_path:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Model Details")
            
            device_obj = torch.device(device)
            model = UNet(in_channels=1, out_channels=1, depth=3, dropout=0.2).to(device_obj)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
            
            # Model type
            st.subheader("🏗️ Architecture")
            st.code("""
UNet Architecture:
- Encoder: 3 levels with 32, 64, 128 channels
- Bottleneck: Double size of final encoder
- Decoder: Mirror of encoder with skip connections
- Final Conv: 1x1 to match output channels
            """, language="text")
        
        with col2:
            st.subheader("⚙️ Configuration")
            
            # Load and display config
            config_path = Path("configs/config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_text = f.read()
                st.code(config_text, language="yaml")
    else:
        st.info("No model selected")
    
    # System info
    st.subheader("💻 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PyTorch Version", torch.__version__)
    with col2:
        st.metric("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
    with col3:
        st.metric("Device", device.upper())

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
