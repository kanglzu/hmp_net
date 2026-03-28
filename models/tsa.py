import torch
import torch.nn as nn
import torch.nn.functional as F

class TSA(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.channels = channels
        
        self.scale_weights = nn.Parameter(torch.ones(3))
        
        self.morph_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 16), channels),
            nn.GELU()
        )
        
        self.topo_encoder = nn.Sequential(
            nn.Conv3d(channels, max(16, channels // 8), 1),
            nn.GroupNorm(max(1, channels // 32), max(16, channels // 8)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv3d(max(16, channels // 8), 3, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(channels + 3, max(8, channels // 16), 1),
            nn.GroupNorm(max(1, channels // 32), max(8, channels // 16)),
            nn.GELU(),
            nn.Conv3d(max(8, channels // 16), 1, 1),
            nn.Sigmoid(),
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
            kernel_size = scale
            
            dilated = F.max_pool3d(x, kernel_size, stride=1, padding=padding)
            eroded = -F.max_pool3d(-x, kernel_size, stride=1, padding=padding)
            morph_grad = dilated - eroded
            
            morph_features.append(morph_grad)
        
        if len(morph_features) > 1:
            weights = F.softmax(self.scale_weights, dim=0)
            combined = sum(w * feat for w, feat in zip(weights, morph_features))
        else:
            combined = morph_features[0]
        
        return combined

    def forward(self, x, return_priors=False):
        identity = x
        B, C, D, H, W = x.shape
        
        morph_feat = self.morph_conv(x)
        multi_scale_grad = self._multi_scale_morphology(morph_feat)
        
        betti_maps = self.topo_encoder(multi_scale_grad)
        
        x_with_topo = torch.cat([x, betti_maps], dim=1)
        spatial_attn = self.spatial_attn(x_with_topo)
        
        attended = x * spatial_attn
        output_feat = self.output_conv(attended)
        
        out = self.gamma * output_feat + self.residual_scale * identity
        
        if return_priors:
            return out, {
                "betti": betti_maps.detach(),
                "spatial_attn": spatial_attn.detach(),
                "scale_weights": self.scale_weights.detach()
            }
        return out
