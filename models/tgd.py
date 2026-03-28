import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologyGuidedDecoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.channels = channels
        
        self.morph_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 16), channels),
            nn.GELU()
        )
        
        self.scale_weights = nn.Parameter(torch.ones(3))
        
        self.topo_guide = nn.Sequential(
            nn.Conv3d(channels, max(8, channels // 16), 1),
            nn.GroupNorm(max(1, channels // 32), max(8, channels // 16)),
            nn.GELU(),
            nn.Conv3d(max(8, channels // 16), 1, 1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(max(1, channels // 16), channels)
        )
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))

    def _multi_scale_morphology(self, x):
        B, C, D, H, W = x.shape
        morph_features = []
        
        for scale in [3, 5, 7]:
            padding = scale // 2
            
            dilated = F.max_pool3d(x, scale, stride=1, padding=padding)
            eroded = -F.max_pool3d(-x, scale, stride=1, padding=padding)
            morph_grad = dilated - eroded
            
            morph_features.append(morph_grad)
        
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * feat for w, feat in zip(weights, morph_features))
        
        return combined

    def forward(self, x):
        identity = x
        B, C, D, H, W = x.shape
        
        morph_feat = self.morph_conv(x)
        multi_scale_grad = self._multi_scale_morphology(morph_feat)
        
        topo_attn = self.topo_guide(multi_scale_grad)
        topo_guided = x * topo_attn
        
        output_feat = self.output_conv(topo_guided)
        out = self.gamma * output_feat + self.residual_scale * identity
        
        return out

    