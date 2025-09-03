import os
import json
import random
import torch
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from collections import Counter

# ==============================
# Config
# ==============================
MODEL_PATH = r"/home/tarik/Projects/Teknofest/Models/MR/best_modela.pt"
TEST_JSON_PATH = r"/home/tarik/Projects/Teknofest/predics/MR_Son.json"
TEST_ROOT_DIR = r"/home/tarik/Projects/Teknofest/Test Dataset/MR_TestSet/MR_testset"
DEPTH = 16
BATCH_SIZE = 2
LABEL_MAP = {"HiperakutAkut": 0, "Subakut": 1, "NormalKronik": 2}
LABEL_NAMES = ['HiperakutAkut', 'Subakut', 'NormalKronik']

# ==============================
# Dataset
# ==============================
class MR3DSliceDataset(Dataset):
    def __init__(self, data_root, json_path, depth=16):
        self.data_root = data_root
        self.depth = depth
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.data = []
        for entry in raw_data:
            if entry.get("Modality") != "MR":
                continue
            pid = str(entry["PatientId"])
            image_id = str(entry["ImageId"])
            lesion = entry.get("LessionName")  
            if lesion not in LABEL_MAP:
                continue
            label = LABEL_MAP[lesion]
            self.data.append((pid, image_id, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid, image_id, label = self.data[idx]
        mr_dir = os.path.join(self.data_root, pid)
        slices = []

        found = False
        current_series = None
        for root, _, files in os.walk(mr_dir):
            if image_id in files:
                current_series = root
                found = True
                break

        if found:
            dcm_files = sorted(f for f in os.listdir(current_series) if f.endswith('.dcm'))
            selected_files = dcm_files[:self.depth]  # eğitim ile aynı: ilk DEPTH slice

            for fname in selected_files:
                try:
                    dcm = pydicom.dcmread(os.path.join(current_series, fname))
                    img = dcm.pixel_array.astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                    img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
                    tensor = torch.from_numpy(img).float() / 255.0
                    slices.append(tensor)
                except:
                    continue

        while len(slices) < self.depth:
            slices.append(torch.zeros((256, 256)))  # eğitim ile aynı: siyah doldurma

        volume = torch.stack(slices).unsqueeze(0)
        return volume, label

# ==============================
# Model (MedNet)
# ==============================
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

# ==============================
# Main Test Function
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = MR3DSliceDataset(TEST_ROOT_DIR, TEST_JSON_PATH, depth=DEPTH)

    class_counts = Counter([lbl for _, _, lbl in dataset.data])
    print("📊 Test veri seti sınıf dağılımı:")
    for k, v in sorted(class_counts.items()):
        print(f"{LABEL_NAMES[k]}: {v}")
    if class_counts.get(1, 0) == 0:
        print("⚠️ Uyarı: Test verisinde Subakut sınıfı bulunmuyor!")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MedNet().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for vols, labels in tqdm(loader, desc="Testing"):
            vols, labels = vols.to(device), labels.to(device)
            outputs = model(vols)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title("Confusion Matrix")
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax in axes:
        idx = random.randint(0, len(dataset)-1)
        vol, label = dataset[idx]
        mid_slice = vol[0, DEPTH//2].numpy()
        ax.imshow(mid_slice, cmap="gray", vmin=0, vmax=1)
        ax.set_title(LABEL_NAMES[label])
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
