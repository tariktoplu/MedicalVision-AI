# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

# --- YENİ MR MODELİ (L.py'den - ConvNeXt-Tiny tabanlı) ---
def MR_ConvNeXt(num_classes=3):
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    return model

# --- ESKİ AMA BT İÇİN DOĞRU OLAN MODEL (Test(2).py'den) ---
class BT_ConvNeXt(nn.Module):
    def __init__(self):
        super(BT_ConvNeXt, self).__init__()
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)