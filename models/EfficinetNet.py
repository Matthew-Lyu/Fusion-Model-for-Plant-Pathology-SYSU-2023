import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x