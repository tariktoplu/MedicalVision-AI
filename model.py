# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

# --- MR MODELİ (Değişiklik Yok) ---
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
        x = self.features(x)
        return self.classifier(x)

# --- BT MODELİ (Doğru Yükleme Mantığı ile Güncellendi) ---
class BT_ConvNeXt(nn.Module):
    def __init__(self):
        super(BT_ConvNeXt, self).__init__()
        
        # 1. Önceden eğitilmiş ConvNeXt'i ORİJİNAL HALİYLE yükle (3 kanallı girdi)
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 2. Sınıflandırıcıyı, eğitimdeki BİREBİR AYNI MİMARİ ile değiştir
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), # İkili sınıflandırma için 1 çıktı
            nn.Sigmoid()
        )
    
    def adapt_grayscale_input(self):
        """
        Bu metot, eğitilmiş 3 kanallı ağırlıklar yüklendikten SONRA çağrılır.
        Modelin giriş katmanını tek kanallı (grayscale) veriye uyarlar.
        """
        # 3 kanallı orijinal katmanın ağırlıklarını al
        original_first_layer = self.model.features[0][0]
        original_weights = original_first_layer.weight.data
        
        # Yeni tek kanallı bir katman oluştur
        new_first_layer = nn.Conv2d(1, original_first_layer.out_channels, 
                                    kernel_size=original_first_layer.kernel_size, 
                                    stride=original_first_layer.stride, 
                                    padding=original_first_layer.padding)
        
        # 3 kanallı ağırlıkların ortalamasını alarak yeni katmana ata
        new_first_layer.weight.data = original_weights.mean(dim=1, keepdim=True)
        if original_first_layer.bias is not None:
            new_first_layer.bias.data = original_first_layer.bias.data
            
        # Orijinal katmanı yeni oluşturduğumuz tek kanallı katman ile değiştir
        self.model.features[0][0] = new_first_layer

    def forward(self, x):
        return self.model(x)