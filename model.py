# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

# --- YENİ MR MODELİ (L.py'deki mantıkla BİREBİR AYNI) ---
def MR_ConvNeXt(num_classes=3):
    """
    L.py'deki build_convnext_tiny fonksiyonunun aynısı.
    Bu bir fonksiyon olduğu için, katman isimleri 'features...' şeklinde olur.
    """
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    return model

# --- BT MODELİ (Değişiklik yok) ---
class BT_ConvNeXt(nn.Module):
    """
    Bu bir sınıf olduğu için, katman isimleri 'model.features...' şeklinde olur.
    Eğitim kodu bu yapıda olduğu için bu şekilde kalmalıdır.
    """
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