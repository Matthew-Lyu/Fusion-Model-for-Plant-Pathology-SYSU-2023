import torch
import torch.nn as nn
import timm
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT


# 消融实验2 resnet+swin+attention
class AttentionModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        weights = self.attention(x)
        return (x * weights).sum(dim=1)


class AwesomeModel3(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        # ResNet
        self.resnet = timm.create_model('resnet50', pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # EfficientNet
        # self.efficientnet = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained)
        # self.efficientnet.classifier= nn.Linear(self.efficientnet.classifier.in_features, num_classes)

        # Swin Transformer
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        self.swin_transformer.head = nn.Linear(self.swin_transformer.head.in_features, num_classes)

        # Attention Module
        self.attention = AttentionModule(num_classes, num_classes)

    def forward(self, x):
        x_resnet = self.resnet(x)
        # x_efficientnet = self.efficientnet(x)
        x_swin = self.swin_transformer(x)
        
        x_stacked = torch.stack([x_resnet,  x_swin], dim=1)

        x_attention = self.attention(x_stacked)

        return x_attention
