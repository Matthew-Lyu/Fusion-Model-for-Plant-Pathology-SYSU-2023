import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x