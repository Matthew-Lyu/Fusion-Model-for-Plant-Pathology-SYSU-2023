import torch
from timm.models.vision_transformer import VisionTransformer

import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ViT, self).__init__()
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            hybrid_backbone=None,
            norm_embed=False,
            pretrained=pretrained
        )

    def forward(self, x):
        x = self.model(x)
        return x
