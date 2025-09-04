# Testt.py

import os
import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import argparse
import csv

# ==============================
# Config
# ==============================
DEPTH = 16
BATCH_SIZE = 4
LABEL_NAMES = ['HiperakutAkut', 'Subakut', 'NormalKronik']

# ==============================
# Dataset (Doğru ve Değişmemiş)
# ==============================
class InferenceDataset(Dataset):
    def __init__(self, data_root, depth=16):
        self.data_root = data_root
        self.depth = depth
        self.file_paths = []
        
        print(f"'{data_root}' klasörü taranıyor...")
        supported_extensions = ('.png', '.jpg', '.jpeg', '.dcm')
        for fname in os.listdir(data_root):
            if fname.lower().endswith(supported_extensions):
                self.file_paths.append(os.path.join(data_root, fname))

        if not self.file_paths:
            raise FileNotFoundError(f"'{data_root}' klasöründe desteklenen formatta görüntü bulunamadı.")
        
        print(f"Toplam {len(self.file_paths)} adet görüntü bulundu.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_extension = os.path.splitext(file_path)[1].lower()
        slices = []

        if file_extension == '.dcm':
            series_dir = os.path.dirname(file_path)
            dcm_files = sorted([f for f in os.listdir(series_dir) if f.lower().endswith('.dcm')])
            selected_files = dcm_files[:self.depth]
            for fname in selected_files:
                try:
                    dcm = pydicom.dcmread(os.path.join(series_dir, fname))
                    img = dcm.pixel_array.astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                    img_resized = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
                    tensor = torch.from_numpy(img_resized).float() / 255.0
                    slices.append(tensor)
                except: continue
        
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            pil_image = Image.open(file_path).convert("L")
            img_array = np.array(pil_image, dtype=np.float32)
            img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6)
            img_resized = np.array(Image.fromarray((img_normalized * 255).astype(np.uint8)).resize((256, 256)))
            tensor_slice = torch.from_numpy(img_resized).float() / 255.0
            slices = [tensor_slice for _ in range(self.depth)]
            
        while len(slices) < self.depth:
            slices.append(torch.zeros((256, 256), dtype=torch.float32))

        volume = torch.stack(slices).unsqueeze(0)
        return volume, file_path

# ==============================
# Model (MedNet - Düzeltilmiş forward metodu ile)
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
        
    # --- DÜZELTME BURADA ---
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x) # Önceki kodda bu satır eksikti!

# ==============================
# Main Inference Function
# ==============================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = InferenceDataset(args.data_dir, depth=DEPTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MedNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = []

    with torch.no_grad():
        for vols, paths in tqdm(loader, desc="Tahmin ediliyor"):
            # DataLoader'dan gelen 'vols' zaten doğru 5D boyutunda.
            vols = vols.to(device)
            outputs = model(vols)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(paths)):
                prediction_index = preds[i].item() # Bu artık hata vermemeli.
                predicted_label = LABEL_NAMES[prediction_index]
                results.append((paths[i], predicted_label))

    output_csv_path = "tahmin_sonuclari.csv"
    print("\n" + "="*50)
    print("TAHMİN SONUÇLARI")
    print("="*50)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['DosyaYolu', 'Tahmin'])
        
        for file_path, prediction in results:
            print(f"- Dosya: {os.path.basename(file_path):<30} -> Tahmin: {prediction}")
            writer.writerow([file_path, prediction])
            
    print("\n" + "="*50)
    print(f"Tüm sonuçlar '{output_csv_path}' dosyasına kaydedildi.")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bir klasördeki MR görüntüleri için tahmin yapar.")
    parser.add_argument("model_path", type=str, help="Eğitilmiş modelin (.pt) yolu.")
    parser.add_argument("data_dir", type=str, help="Tahmin yapılacak görüntülerin bulunduğu klasör.")
    
    args = parser.parse_args()
    main(args)