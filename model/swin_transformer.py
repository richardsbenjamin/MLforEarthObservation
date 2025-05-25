import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinTransformerV1(nn.Module):
    def __init__(self, img_size=30, in_chans=1, out_chans=1):
        super().__init__()
        
        # Configuração básica do Swin Transformer
        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            num_classes=0,  # Sem camada de classificação
            embed_dim=96,    # Dimensão menor para simplificar
            depths=[2, 2],   # Menos camadas
            num_heads=[3, 6],# Menos cabeças de atenção
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True
        )
        
        # Camada final para ajustar o output
        self.final_conv = nn.Conv2d(96, out_chans, kernel_size=1)
        
    def forward(self, x):
        # Garante que a entrada tem 4 dimensões (batch, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Extrai features com Swin Transformer
        features = self.swin.forward_features(x)
        
        # Redimensiona as features para formato de imagem
        batch_size = x.shape[0]
        h = w = int(features.shape[1] ** 0.5)  # Assume features são quadradas
        features = features.view(batch_size, h, w, -1).permute(0, 4, 1, 2)
        
        # Aplica convolução final
        output = self.final_conv(features)
        
        return output