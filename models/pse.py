import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, stride=1):
        super().__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size, stride, 
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, kernel_size, 1, 
                     padding=kernel_size//2, groups=init_channels, bias=False),
            nn.BatchNorm3d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class PSE(nn.Module):
    def __init__(self, in_channels=32, out_channels=64, num_modalities=4):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.modality_encoders = nn.ModuleList([
            GhostConv3D(in_channels, in_channels, ratio=2) 
            for _ in range(num_modalities)
        ])
        
        coupling_init = torch.tensor([
            [1.0, 0.9, 0.3, 0.3],
            [0.9, 1.0, 0.4, 0.4],
            [0.3, 0.4, 1.0, 0.8],
            [0.3, 0.4, 0.8, 1.0],
        ])
        self.coupling_matrix = nn.Parameter(coupling_init)
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels * num_modalities, 8, 1),
            nn.ReLU(),
            nn.Conv3d(8, in_channels * num_modalities, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * num_modalities, out_channels, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x, return_priors=False):
        B, M, C, D, H, W = x.shape
        assert M == self.num_modalities, f"Expected {self.num_modalities} modalities, got {M}"
        
        encoded = []
        for m in range(M):
            modality_feat = self.modality_encoders[m](x[:, m])
            encoded.append(modality_feat)
        
        encoded = torch.stack(encoded, dim=1)
        coupling = F.softmax(self.coupling_matrix, dim=1)
        
        encoded_flat = encoded.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, M)
        coupled = torch.bmm(encoded_flat, coupling.unsqueeze(0).expand(B, -1, -1))
        coupled = coupled.reshape(B, C, D, H, W, M).permute(0, 5, 1, 2, 3, 4)
        
        concat = coupled.reshape(B, M * C, D, H, W)
        attn = self.channel_attn(concat)
        refined = concat * attn
        
        out = self.fusion(refined)
        
        if return_priors:
            return out, {"coupling": coupling.detach()}
        return out

