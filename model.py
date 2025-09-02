# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

# --- MR MODELİ (mednet_newest.py'den) ---
class MedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 64 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # MedNet'in forward fonksiyonu Conv3D beklediği için gelen (1, 1, D, H, W) tensörünü
        # (1, D, H, W) şekline getirmemiz gerekebilir. Testt.py'de bu (1, D, H, W) olarak veriliyor.
        # Bu yüzden gelen verinin doğruluğundan emin olmalıyız. 
        # `workers.py`'deki `unsqueeze(0)`'ı kaldırdık.
        x = self.features(x)
        return self.classifier(x)

# --- BT MODELİ (ConvNext_Tiny.py ve Test.py'den) ---
class BT_ConvNeXt(nn.Module):
    def __init__(self):
        super(BT_ConvNeXt, self).__init__()
        # Eğitimde olduğu gibi, orijinal 3 kanallı modeli yüklüyoruz.
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # Sınıflandırıcıyı, eğitimdeki BİREBİR AYNI MİMARİ ile değiştiriyoruz.
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), # İkili sınıflandırma
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)