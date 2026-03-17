import torch
import torch.nn as nn


class UNet(nn.Module):
    """Lightweight UNet for medical image enhancement with low data constraints."""
    
    def __init__(self, in_channels=1, out_channels=1, depth=3, dropout=0.2):
        super(UNet, self).__init__()
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
    
    def forward(self, x):
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
    
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(ResidualBlock, self).__init__()
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
    
    def forward(self, x):
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
    
    def __init__(self, in_channels=1, out_channels=1, depth=3, dropout=0.2):
        super(ResUNet, self).__init__()
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
    
    def forward(self, x):
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
    
    def __init__(self, in_channels, growth_rate=16, num_layers=4, dropout=0.2):
        super(DenseBlock, self).__init__()
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
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


def get_model(architecture='UNet', in_channels=1, out_channels=1, depth=3, 
              dropout=0.2, **kwargs) -> nn.Module:
    """Factory function to get model based on architecture name."""
    
    if architecture.lower() == 'unet':
        return UNet(in_channels, out_channels, depth, dropout)
    elif architecture.lower() == 'resunet':
        return ResUNet(in_channels, out_channels, depth, dropout)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Expert models for specific tasks
class XrayEnhancementNet(nn.Module):
    """Specialized network for X-ray enhancement with multi-scale processing."""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(XrayEnhancementNet, self).__init__()
        
        # Multi-scale encoder
        self.scale1 = self._conv_block(in_channels, 32)
        self.scale2 = nn.Sequential(nn.MaxPool2d(2, 2), self._conv_block(in_channels, 32))
        
        # Main UNet
        self.unet = UNet(in_channels, out_channels, depth=3, dropout=0.2)
        
        # Fusion
        self.fusion = self._conv_block(32 + out_channels, out_channels)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Main enhancement
        enhanced = self.unet(x)
        
        # Multi-scale features
        feat1 = self.scale1(x)
        
        # Combine
        combined = torch.cat([feat1, enhanced], dim=1)
        output = self.fusion(combined)
        
        return output
