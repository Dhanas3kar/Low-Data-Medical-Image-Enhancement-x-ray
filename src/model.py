import torch
import torch.nn as nn
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class UNet(nn.Module):
    """Lightweight UNet for medical image enhancement with low data constraints."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, depth: int = 3, dropout: float = 0.2):
        super(UNet, self).__init__()
        
        # Input validation
        if not isinstance(depth, int) or depth < 1 or depth > 5:
            raise ValueError(f"depth must be an integer between 1 and 5, got {depth}")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError(f"in_channels must be positive integer, got {in_channels}")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError(f"out_channels must be positive integer, got {out_channels}")
        
        self.depth = depth
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        channels = [in_channels, 32, 64, 128]
        for i in range(depth):
            self.encoder.append(
                self._double_conv(channels[i], channels[i+1], dropout)
            )
        
        # Bottleneck
        self.bottleneck = self._double_conv(channels[depth], channels[depth] * 2, dropout)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.up = nn.ModuleList()
        
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[i] * 2, channels[i], 2, 2))
            self.decoder.append(
                self._double_conv(channels[i] * 2, channels[i], dropout)
            )
        
        # Final output layer
        self.final = nn.Conv2d(channels[1], out_channels, 1)
    
    def _double_conv(self, in_ch, out_ch, dropout=0.2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input validation and robustness."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D tensor")
        
        # Ensure 4D tensor (B,C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        encoder_outputs = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.up, self.decoder)):
            x = up(x)
            skip = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        
        # Final layer
        x = self.final(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with option for dense connections."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError(f"in_channels must be positive integer, got {in_channels}")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError(f"out_channels must be positive integer, got {out_channels}")
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    """UNet with residual blocks for enhanced feature learning."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, depth: int = 3, dropout: float = 0.2):
        super(ResUNet, self).__init__()
        
        if not isinstance(depth, int) or depth < 1 or depth > 5:
            raise ValueError(f"depth must be between 1 and 5, got {depth}")
        
        self.depth = depth
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        channels = [in_channels, 32, 64, 128]
        for i in range(depth):
            self.encoder.append(
                nn.Sequential(
                    ResidualBlock(channels[i], channels[i+1], dropout),
                    ResidualBlock(channels[i+1], channels[i+1], dropout)
                )
            )
        
        # Bottleneck
        self.bottleneck = ResidualBlock(channels[depth], channels[depth] * 2, dropout)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.up = nn.ModuleList()
        
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[i] * 2, channels[i], 2, 2))
            self.decoder.append(
                nn.Sequential(
                    ResidualBlock(channels[i] * 2, channels[i], dropout),
                    ResidualBlock(channels[i], channels[i], dropout)
                )
            )
        
        # Final output layer
        self.final = nn.Conv2d(channels[1], out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input validation."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        encoder_outputs = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.up, self.decoder)):
            x = up(x)
            skip = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        
        # Final layer
        x = self.final(x)
        return x


class DenseBlock(nn.Module):
    """Dense block with concatenated feature maps."""
    
    def __init__(self, in_channels: int, growth_rate: int = 16, num_layers: int = 4, dropout: float = 0.2):
        super(DenseBlock, self).__init__()
        
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be positive integer, got {num_layers}")
        if not isinstance(growth_rate, int) or growth_rate < 1:
            raise ValueError(f"growth_rate must be positive integer, got {growth_rate}")
        
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        
        for i in range(num_layers):
            layer_in = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(layer_in),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(layer_in, growth_rate, 3, padding=1),
                    nn.Dropout2d(dropout)
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feature concatenation."""
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


def get_model(architecture: str = 'UNet', in_channels: int = 1, out_channels: int = 1, depth: int = 3, 
              dropout: float = 0.2, **kwargs) -> nn.Module:
    """Factory function to get model based on architecture name with validation."""
    
    if not isinstance(architecture, str):
        raise TypeError(f"architecture must be string, got {type(architecture)}")
    
    arch_lower = architecture.lower()
    
    if arch_lower == 'unet':
        return UNet(in_channels, out_channels, depth, dropout)
    elif arch_lower == 'resunet':
        return ResUNet(in_channels, out_channels, depth, dropout)
    elif arch_lower == 'xrayenhancementnet':
        return XrayEnhancementNet(in_channels, out_channels)
    else:
        available = ['unet', 'resunet', 'xrayenhancementnet']
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")


# Expert models for specific tasks
class XrayEnhancementNet(nn.Module):
    """Specialized network for X-ray enhancement with multi-scale processing."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(XrayEnhancementNet, self).__init__()
        
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError(f"in_channels must be positive integer, got {in_channels}")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError(f"out_channels must be positive integer, got {out_channels}")
        
        # Multi-scale encoder
        self.scale1 = self._conv_block(in_channels, 32)
        self.scale2 = nn.Sequential(nn.MaxPool2d(2, 2), self._conv_block(in_channels, 32))
        
        # Main UNet
        self.unet = UNet(in_channels, out_channels, depth=3, dropout=0.2)
        
        # Fusion: expects concatenated features
        self.fusion = self._conv_block(32 + out_channels, out_channels)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create a simple convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale enhancement."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        # Main enhancement
        enhanced = self.unet(x)
        
        # Multi-scale features
        feat1 = self.scale1(x)
        
        # Combine features
        combined = torch.cat([feat1, enhanced], dim=1)
        output = self.fusion(combined)
        
        return output
