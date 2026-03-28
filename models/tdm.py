import torch
import torch.nn as nn
import torch.nn.functional as F

class TDM(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.channels = channels
        self.dt = nn.Parameter(torch.tensor(0.1))
        
        self.physio_predictor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
        self.laplacian_conv = nn.Conv3d(channels, channels, 3, 
                                      padding=1, groups=channels, bias=False)
        self._init_laplacian_weights()
        
        self.spatial_weight = nn.Sequential(
            nn.Conv3d(channels, max(8, channels // 16), 1),
            nn.GroupNorm(max(1, channels // 32), max(8, channels // 16)),
            nn.GELU(),
            nn.Conv3d(max(8, channels // 16), 1, 1),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(max(1, channels // 16), channels)
        )
        
        self.gamma = nn.Parameter(torch.ones(1))

    def _init_laplacian_weights(self):
        weight = torch.zeros(1, 1, 3, 3, 3)
        weight[0, 0, 1, 1, 1] = 6.0
        for d in range(3):
            for h in range(3):
                for w in range(3):
                    if (d, h, w) != (1, 1, 1):
                        weight[0, 0, d, h, w] = -1.0
        weight = weight / 6.0
        
        with torch.no_grad():
            self.laplacian_conv.weight.data = weight.repeat(
                self.channels, 1, 1, 1, 1)

    def forward(self, x, return_priors=False):
        B, C, D, H, W = x.shape
        identity = x
        
        physio = self.physio_predictor(x)
        rho = physio[:, 0].view(B, 1, 1, 1, 1)
        D_val = physio[:, 1].view(B, 1, 1, 1, 1)
        
        laplacian = self.laplacian_conv(x) - x
        u_norm = torch.sigmoid(x)
        reaction = u_norm * (1 - u_norm)
        
        dt_clamped = torch.clamp(self.dt, 0.01, 0.5)
        evolved = x + dt_clamped * (D_val * laplacian + rho * reaction)
        
        spatial_w = self.spatial_weight(evolved)
        modulated = x + spatial_w * (evolved - x)
        
        out_proj = self.output_proj(modulated)
        out = self.gamma * out_proj + identity
        
        if return_priors:
            return out, {
                "rho": rho.detach(),
                "D": D_val.detach(),
                "dt": dt_clamped.detach()
            }
        return out
