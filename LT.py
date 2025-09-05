# file: test_inference_convnext.py
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pydicom
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix

import torchvision.transforms as T
import torchvision.models as models

# -----------------------------
# Yol ayarları
# -----------------------------
# Lütfen bu yolları kendi sisteminize göre güncelleyin
MODEL_PATH = "/home/tarik/Projects/Teknofest/Models/MR/v2.pt" # Örnek olarak v2 kullanıldı
TEST_JSON_PATH = "/home/tarik/Projects/Teknofest/Test Dataset/MR_TestSet/MR_testset/MR_Son.json"
TEST_ROOT_DIR = "/home/tarik/Projects/Teknofest/Test Dataset/MR_TestSet/MR_testset"

LABEL_MAP = {"HiperakutAkut": 0, "Subakut": 1, "NormalKronik": 2}
IDX2NAME = {v: k for k, v in LABEL_MAP.items()}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -----------------------------
# Yardımcılar
# -----------------------------
def robust_dicom_to_pil_rgb(path: str) -> Image.Image:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo: lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    arr = (arr - lo) / denom
    img = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode='L').convert('RGB')

def make_transform(img_size=224):
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_convnext_tiny(num_classes=3):
    m = models.convnext_tiny(weights=None)
    in_feats = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_feats, num_classes)
    return m

# -----------------------------
# Dataset
# -----------------------------
class MRDicomJsonDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        pid, image_id, label, full_path = self.items[idx]
        img = robust_dicom_to_pil_rgb(full_path)
        if self.transform: img = self.transform(img)
        return img, label

def build_patient_file_index(root, patient_ids):
    index = {}
    for pid in sorted(set(patient_ids)):
        filemap = {}
        for base in [os.path.join(root, pid, "MR"), os.path.join(root, pid)]:
            if not os.path.isdir(base): continue
            for r, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith(".dcm"):
                        filemap[f] = os.path.join(r, f)
        index[pid] = filemap
    return index

def load_items(json_path, root):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items_raw = []
    for e in data:
        if str(e.get("Modality","")).upper() != "MR": continue
        lesion = e.get("LessionName")
        if lesion not in LABEL_MAP: continue
        pid = str(e["PatientId"])
        image_id = str(e["ImageId"])
        label = LABEL_MAP[lesion]
        items_raw.append((pid, image_id, label))
    # path bağla
    byp = {}
    for pid, image_id, _ in items_raw:
        byp.setdefault(pid,set()).add(image_id)
    index = build_patient_file_index(root, list(byp.keys()))
    items = []
    for pid, image_id, label in items_raw:
        p = index.get(pid,{}).get(image_id)
        if p: items.append((pid, image_id, label, p))
    return items

# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Test set hazırlanıyor...")
    test_items = load_items(TEST_JSON_PATH, TEST_ROOT_DIR)
    print(f"Test set boyutu: {len(test_items)}")

    transform = make_transform(224)
    ds_test = MRDicomJsonDataset(test_items, transform)
    dl_test = DataLoader(ds_test, batch_size=16, shuffle=False)

    # Model yükle
    model = build_convnext_tiny(num_classes=3).to(device)
    
    # --- DÜZELTME BURADA ---
    # Yeni PyTorch versiyonlarının güvenlik kontrolünü geçmek için weights_only=False ekliyoruz.
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model yüklendi: {MODEL_PATH}")

    # Test et
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in dl_test:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    print("\n=== Test Sonuçları ===")
    print(classification_report(y_true, y_pred,
                                target_names=[IDX2NAME[i] for i in range(3)],
                                digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()