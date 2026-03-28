import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsGuidedDecoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        
        self.physics_guide = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv3d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_modulation = nn.Sequential(
            nn.Conv3d(channels, channels // 4, 1),
            nn.GroupNorm(max(1, channels // 32), channels // 4),
            nn.GELU(),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(max(1, channels // 8), channels)
        )
        
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        
        physics_attn = self.physics_guide(x)
        physics_guided = x * physics_attn
        
        spatial_attn = self.spatial_modulation(physics_guided)
        modulated = physics_guided * spatial_attn
        
        out = self.output_conv(modulated)
        out = self.gamma * out + identity
        
        return out