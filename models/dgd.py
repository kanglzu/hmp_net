import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsGuidedDecoder(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        self.dt = nn.Parameter(torch.tensor(0.05))
        
        self.dynamics_guide = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(16, channels // 16), 1),
            nn.GELU(),
            nn.Conv3d(max(16, channels // 16), 1, 1),
            nn.Sigmoid()
        )
        
        self.diffusion_conv = nn.Conv3d(channels, channels, 3, 
                                       padding=1, groups=channels, bias=False)
        self._init_diffusion_weights()
        
        self.spatial_weight = nn.Sequential(
            nn.Conv3d(channels, max(8, channels // 32), 1),
            nn.GroupNorm(max(1, channels // 64), max(8, channels // 32)),
            nn.GELU(),
            nn.Conv3d(max(8, channels // 32), 1, 1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(max(1, channels // 32), channels)
        )
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.diffusion_scale = nn.Parameter(torch.ones(1))

    def _init_diffusion_weights(self):
        weight = torch.zeros(1, 1, 3, 3, 3)
        weight[0, 0, 1, 1, 1] = 6.0
        for d in range(3):
            for h in range(3):
                for w in range(3):
                    if (d, h, w) != (1, 1, 1):
                        weight[0, 0, d, h, w] = -1.0
        weight = weight / 6.0
        
        with torch.no_grad():
            self.diffusion_conv.weight.data = weight.repeat(
                self.channels, 1, 1, 1, 1)

    def forward(self, x, return_priors=False):
        identity = x
        B, C, D, H, W = x.shape
        
        dynamics_w = self.dynamics_guide(x)
        diffusion_term = self.diffusion_conv(x) - x
        
        dt_clamped = torch.clamp(self.dt, 0.01, 0.1)
        evolved = x + dt_clamped * dynamics_w * self.diffusion_scale * diffusion_term
        
        spatial_w = self.spatial_weight(x)
        modulated = x + spatial_w * (evolved - x)
        
        output_feat = self.output_conv(modulated)
        out = self.gamma * output_feat + self.residual_scale * identity
        
        if return_priors:
            return out, {
                "dynamics_weight": dynamics_w.detach(),
                "spatial_weight": spatial_w.detach(),
                "dt": dt_clamped.detach(),
                "diffusion_scale": self.diffusion_scale.detach()
            }
        return out

