import torch
import torch.nn as nn
import torch.nn.functional as F

from .pse import PSE
from .tsa import TSA  
from .tdm import TDM
from .pgd import PhysicsGuidedDecoder
from .tgd import TopologyGuidedDecoder
from .dgd import DynamicsGuidedDecoder
from .emp_skip import EnhancedMPSkipConnection

class CrossModalFusion(nn.Module):
    def __init__(self, num_modalities=4, out_channels=32):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 8, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, 8), 8),
                nn.GELU()
            ) for _ in range(num_modalities)
        ])
        self.cross_modal_attn = nn.Sequential(
            nn.Conv3d(num_modalities * 8, max(8, num_modalities * 2), 1),
            nn.GroupNorm(max(1, num_modalities * 2), max(8, num_modalities * 2)),
            nn.GELU(),
            nn.Conv3d(max(8, num_modalities * 2), num_modalities, 1),
            nn.Sigmoid()
        )
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        self.output_proj = nn.Sequential(
            nn.Conv3d(num_modalities * 8, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, out_channels // 4), out_channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, M, D, H, W = x.shape
        modality_feats = []
        for m in range(M):
            feat = self.modality_encoders[m](x[:, m:m+1])
            modality_feats.append(feat)
        concat_feats = torch.cat(modality_feats, dim=1)
        attn = self.cross_modal_attn(concat_feats)
        weighted_feats = []
        for m in range(M):
            weight = self.modality_weights[m] * attn[:, m:m+1]
            weighted_feats.append(modality_feats[m] * weight)
        fused = torch.cat(weighted_feats, dim=1)
        output = self.output_proj(fused)
        return self.gamma * output

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 8), out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 8), out_channels),
            nn.GELU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(max(1, out_channels // 8), out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.gamma * x + identity
        return F.gelu(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 8), out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 8), out_channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.gamma * x

class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv3d(in_channels, max(8, in_channels // 2), 3, padding=1, bias=False),
            nn.GroupNorm(max(1, in_channels // 16), max(8, in_channels // 2)),
            nn.GELU(),
            nn.Conv3d(max(8, in_channels // 2), num_classes, 1)
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, target_size=None):
        logits = self.classifier(x)
        logits = self.gamma * logits
        if target_size is not None:
            logits = F.interpolate(logits, size=target_size, mode='trilinear', align_corners=False)
        return logits

class HMPNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_channels=32, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        enc_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        dec_channels = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels]
        
        self.cmf = CrossModalFusion(num_modalities=in_channels, out_channels=enc_channels[0])
        self.stem = nn.Sequential(
            nn.Conv3d(enc_channels[0], enc_channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(max(1, enc_channels[0] // 8), enc_channels[0]),
            nn.GELU()
        )
        
        self.pse = PSE(in_channels=enc_channels[0], out_channels=enc_channels[1])
        self.encoder1 = EncoderBlock(enc_channels[1], enc_channels[1])
        self.down1 = nn.Sequential(
            nn.Conv3d(enc_channels[1], enc_channels[2], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, enc_channels[2] // 8), enc_channels[2]),
            nn.GELU()
        )
        
        self.tsa = TSA(channels=enc_channels[2])
        self.encoder2 = EncoderBlock(enc_channels[2], enc_channels[2])
        self.down2 = nn.Sequential(
            nn.Conv3d(enc_channels[2], enc_channels[3], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, enc_channels[3] // 16), enc_channels[3]),
            nn.GELU()
        )
        
        self.encoder3 = EncoderBlock(enc_channels[3], enc_channels[3])
        self.down3 = nn.Sequential(
            nn.Conv3d(enc_channels[3], enc_channels[3], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, enc_channels[3] // 16), enc_channels[3]),
            nn.GELU()
        )
        
        self.encoder4 = EncoderBlock(enc_channels[3], enc_channels[3])
        self.tdm = TDM(channels=enc_channels[3])
        self.bottleneck = EncoderBlock(enc_channels[3], enc_channels[3])
        
        self.up4 = nn.ConvTranspose3d(enc_channels[3], dec_channels[1], 2, stride=2)
        self.skip4 = EnhancedMPSkipConnection(encoder_channels=enc_channels[3], decoder_channels=dec_channels[1], prior_type=None)
        self.decoder4 = DecoderBlock(dec_channels[1], dec_channels[1])
        self.dgd = DynamicsGuidedDecoder(channels=dec_channels[1])
        
        self.up3 = nn.ConvTranspose3d(dec_channels[1], enc_channels[2], 2, stride=2)
        self.skip3 = EnhancedMPSkipConnection(encoder_channels=enc_channels[3], decoder_channels=dec_channels[1], prior_type='topology')
        self.decoder3 = DecoderBlock(enc_channels[2], enc_channels[2])
        self.tgd = TopologyGuidedDecoder(channels=enc_channels[2])
        
        self.up2 = nn.ConvTranspose3d(enc_channels[2], enc_channels[1], 2, stride=2)
        self.skip2 = EnhancedMPSkipConnection(encoder_channels=enc_channels[2], decoder_channels=enc_channels[1], prior_type='physics')
        self.decoder2 = DecoderBlock(enc_channels[1], enc_channels[1])
        self.pgd = PhysicsGuidedDecoder(channels=enc_channels[1])
        
        if deep_supervision:
            self.aux_head1 = DeepSupervisionHead(dec_channels[1], num_classes)
            self.aux_head2 = DeepSupervisionHead(enc_channels[2], num_classes)
            self.aux_head3 = DeepSupervisionHead(enc_channels[1], num_classes)
        
        self.final_head = DeepSupervisionHead(dec_channels[2], num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_aux=None, return_priors=False):
        if return_aux is None:
            return_aux = self.training and self.deep_supervision
        
        B, M, D, H, W = x.shape
        target_size = (D, H, W)
        
        x_fused = self.cmf(x)
        x_stem = self.stem(x_fused)
        x_multi = torch.stack([x_stem] * M, dim=1)
        
        if return_priors:
            enc1, pse_priors = self.pse(x_multi, return_priors=True)
        else:
            enc1 = self.pse(x_multi)
        enc1 = self.encoder1(enc1)
        
        x = self.down1(enc1)
        if return_priors:
            enc2, tsa_priors = self.tsa(x, return_priors=True)
        else:
            enc2 = self.tsa(x)
        enc2 = self.encoder2(enc2)
        
        x = self.down2(enc2)
        enc3 = self.encoder3(x)
        
        x = self.down3(enc3)
        enc4 = self.encoder4(x)
        if return_priors:
            enc4, tdm_priors = self.tdm(enc4, return_priors=True)
        else:
            enc4 = self.tdm(enc4)
        bottleneck = self.bottleneck(enc4)
        
        aux_outputs = []
        
        x = self.up4(bottleneck)
        x = self.skip4(enc4, x)
        dec4 = self.decoder4(x)
        dec4 = self.dgd(dec4)
        if return_aux:
            aux1 = self.aux_head1(dec4, target_size)
            aux_outputs.append(aux1)
        
        x = self.up3(dec4)
        x = self.skip3(enc3, x)
        dec3 = self.decoder3(x)
        dec3 = self.tgd(dec3)
        if return_aux:
            aux2 = self.aux_head2(dec3, target_size)
            aux_outputs.append(aux2)
        
        x = self.up2(dec3)
        x = self.skip2(enc2, x)
        dec2 = self.decoder2(x)
        dec2 = self.pgd(dec2)
        if return_aux:
            aux3 = self.aux_head3(dec2, target_size)
            aux_outputs.append(aux3)
        
        final = self.final_head(dec2, target_size)
        
        if return_aux or return_priors:
            result = [final]
            if return_aux:
                result.append(aux_outputs)
            if return_priors:
                priors_dict = {}
                if 'pse_priors' in locals():
                    priors_dict['pse'] = pse_priors
                if 'tsa_priors' in locals():
                    priors_dict['tsa'] = tsa_priors  
                if 'tdm_priors' in locals():
                    priors_dict['tdm'] = tdm_priors
                result.append(priors_dict)
            return tuple(result)
        else:
            return final

def build_hmpnet(config=None):
    if config is None:
        config = {}
    model = HMPNet(
        in_channels=config.get('in_channels', 4),
        num_classes=config.get('num_classes', 4),
        base_channels=config.get('base_channels', 32),
        deep_supervision=config.get('deep_supervision', True)
    )
    return model

