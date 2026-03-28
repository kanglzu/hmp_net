import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMPSkipConnection(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, prior_type=None):
        super().__init__()
        self.prior_type = prior_type
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        self.align = nn.Sequential(
            nn.Conv3d(encoder_channels, decoder_channels, 1, bias=False),
            nn.GroupNorm(max(1, decoder_channels // 16), decoder_channels),
            nn.GELU()
        )
        
        if prior_type == 'physics':
            self.prior_guide = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(decoder_channels, max(8, decoder_channels // 16), 1),
                nn.GELU(),
                nn.Conv3d(max(8, decoder_channels // 16), decoder_channels, 1),
                nn.Sigmoid()
            )
        elif prior_type == 'topology':
            self.prior_guide = nn.Sequential(
                nn.Conv3d(decoder_channels, max(8, decoder_channels // 16), 1),
                nn.GroupNorm(max(1, decoder_channels // 32), max(8, decoder_channels // 16)),
                nn.GELU(),
                nn.Conv3d(max(8, decoder_channels // 16), 1, 1),
                nn.Sigmoid()
            )
        elif prior_type == 'dynamics':
            self.global_guide = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(decoder_channels, max(8, decoder_channels // 16), 1),
                nn.GELU(),
                nn.Conv3d(max(8, decoder_channels // 16), decoder_channels, 1),
                nn.Sigmoid()
            )
            self.spatial_guide = nn.Sequential(
                nn.Conv3d(decoder_channels, max(8, decoder_channels // 16), 1),
                nn.GroupNorm(max(1, decoder_channels // 32), max(8, decoder_channels // 16)),
                nn.GELU(),
                nn.Conv3d(max(8, decoder_channels // 16), 1, 1),
                nn.Sigmoid()
            )
            self.mix_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.prior_guide = None
            self.global_guide = None
            self.spatial_guide = None
        
        self.cross_attn = nn.Sequential(
            nn.Conv3d(decoder_channels * 2, max(16, decoder_channels // 8), 1),
            nn.GroupNorm(max(1, decoder_channels // 32), max(16, decoder_channels // 8)),
            nn.GELU(),
            nn.Conv3d(max(16, decoder_channels // 8), decoder_channels, 1),
            nn.Sigmoid()
        )
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, encoder_feat, decoder_feat):
        B, C_enc, D_enc, H_enc, W_enc = encoder_feat.shape
        B, C_dec, D_dec, H_dec, W_dec = decoder_feat.shape
        
        if (D_enc, H_enc, W_enc) != (D_dec, H_dec, W_dec):
            encoder_feat = F.interpolate(
                encoder_feat, 
                size=(D_dec, H_dec, W_dec), 
                mode='trilinear', 
                align_corners=False
            )
        
        encoder_aligned = self.align(encoder_feat)
        
        if self.prior_type == 'dynamics':
            global_attn = self.global_guide(decoder_feat)
            spatial_attn = self.spatial_guide(decoder_feat)
            
            mix_weight = torch.sigmoid(self.mix_weight)
            prior_attn = mix_weight * global_attn + (1 - mix_weight) * spatial_attn
            
            decoder_guided = decoder_feat * prior_attn
            
        elif self.prior_guide is not None:
            prior_attn = self.prior_guide(decoder_feat)
            decoder_guided = decoder_feat * prior_attn
        else:
            decoder_guided = decoder_feat
        
        concat_feat = torch.cat([encoder_aligned, decoder_guided], dim=1)
        cross_attention = self.cross_attn(concat_feat)
        
        attended_encoder = encoder_aligned * cross_attention
        output = self.gamma * attended_encoder + self.residual_scale * decoder_guided
        
        return output
