import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=pretrained)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x
        