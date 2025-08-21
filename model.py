# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

# --- MR MODELİ (mednet.py dosyasından alındı - Değişiklik Yok) ---

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

# --- BT MODELİ (Hata mesajına göre tamamen yeniden düzenlendi) ---

class BT_ConvNeXt(nn.Module):
    def __init__(self, num_classes=3):
        super(BT_ConvNeXt, self).__init__()
        
        # 1. Adım: Önceden eğitilmiş ConvNeXt modelini olduğu gibi yükle.
        # Bu model 3 kanallı girdi bekler.
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 2. Adım: Sınıflandırıcıyı, eğitim kodundaki (ConvNext_Tiny.py)
        # BİREBİR AYNI MİMARİ ile değiştir.
        # Bu, 1 sınıflı çıktı verir ve ikili sınıflandırma için kullanılır.
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), # DİKKAT: Eğitildiği gibi 1 çıktılı
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

    def adapt_to_application(self):
        """
        Bu metot, eğitilmiş ağırlıklar yüklendikten SONRA çağrılır.
        Modeli tek kanallı girdi ve çok sınıflı çıktıya uyarlar.
        """
        # --- GİRİŞ KATMANINI UYARLAMA ---
        # 3 kanallı eğitilmiş ağırlıkları alıp ortalamasını alarak
        # tek kanallı yeni bir katman oluşturuyoruz.
        original_first_layer = self.model.features[0][0]
        original_weights = original_first_layer.weight.data
        
        new_first_layer = nn.Conv2d(1, original_first_layer.out_channels, 
                                    kernel_size=original_first_layer.kernel_size, 
                                    stride=original_first_layer.stride, 
                                    padding=original_first_layer.padding)
        
        new_first_layer.weight.data = original_weights.mean(dim=1, keepdim=True)
        if original_first_layer.bias is not None:
            new_first_layer.bias.data = original_first_layer.bias.data
            
        self.model.features[0][0] = new_first_layer
        
        # --- ÇIKIŞ KATMANINI UYARLAMA ---
        # Sınıflandırıcının sonundaki ikili (binary) katmanları söküp,
        # yerine 3 sınıflı (multiclass) yeni bir katman takıyoruz.
        num_ftrs_before_last = self.model.classifier[2][4].in_features # nn.Linear(128, 1) katmanının giriş boyutu
        
        # Eski sınıflandırıcının son iki katmanını (Linear(128,1) ve Sigmoid) at
        self.model.classifier[2] = self.model.classifier[2][:-2]
        
        # Yeni çok sınıflı katmanı ekle
        self.model.classifier[2].add_module("4", nn.Linear(num_ftrs_before_last, 3))