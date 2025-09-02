# workers.py

import os
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from PyQt5.QtCore import QThread, pyqtSignal

# --- BT için Test.py'den alınan DICOM işleme yardımcı fonksiyonları ---
def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=0.0)
    vmin, vmax = np.percentile(img, [0.5, 99.5]) if np.ptp(img) > 0 else (img.min(), img.max())
    if vmax == vmin:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return (img * 255.0).round().astype(np.uint8)

def _window_image(ds: pydicom.dataset.FileDataset, default_center: float = 40.0, default_width: float = 80.0) -> np.ndarray:
    data = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    data = data * slope + intercept
    try:
        data = apply_voi_lut(data, ds)
        return _to_uint8(data)
    except Exception:
        pass
    center = float(ds.get('WindowCenter', default_center))
    width = float(ds.get('WindowWidth', default_width))
    if isinstance(center, pydicom.multival.MultiValue): center = float(center[0])
    if isinstance(width, pydicom.multival.MultiValue): width = float(width[0])
    low = center - width / 2.0
    high = center + width / 2.0
    data = np.clip(data, low, high)
    return _to_uint8(data)

class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)
    
    def __init__(self, models, device, file_path, label_names, modality):
        super().__init__()
        self.models = models
        self.device = device
        self.file_path = file_path
        self.label_names = label_names
        self.modality = modality
        
    def run(self):
        try:
            self.progress.emit("Görüntü işleniyor...")
            image_tensor = self.preprocess_image(self.file_path)
            if image_tensor is None:
                raise Exception("Görüntü işlenemedi veya desteklenmiyor.")
                
            image_tensor_gpu = image_tensor.to(self.device)
            self.progress.emit("Model analizi yapılıyor...")
            
            if self.modality == 'MR':
                predictions = []
                with torch.no_grad():
                    for model in self.models:
                        output = model(image_tensor_gpu)
                        pred_probs = torch.softmax(output, dim=1)
                        predictions.append(pred_probs.cpu().numpy()[0])
                avg_prediction = np.mean(predictions, axis=0)
                predicted_class_idx = np.argmax(avg_prediction)
                all_probabilities = (avg_prediction * 100).tolist()
            
            elif self.modality == 'BT':
                outputs = []
                with torch.no_grad():
                    for model in self.models:
                        output = model(image_tensor_gpu)
                        outputs.append(output.cpu().numpy()[0][0])
                
                avg_prob_stroke = np.mean(outputs)
                all_probabilities_raw = [1 - avg_prob_stroke, avg_prob_stroke]
                predicted_class_idx = np.argmax(all_probabilities_raw)
                all_probabilities = [p * 100 for p in all_probabilities_raw]

            predicted_label = self.label_names[predicted_class_idx]
            self.finished.emit(predicted_label, all_probabilities)
            
        except Exception as e:
            self.error.emit(f"Analiz hatası: {str(e)}")
    
    def preprocess_image(self, file_path):
        try:
            if self.modality == 'MR':
                # --- Testt.py'deki MANTIĞIN BİREBİR UYGULANMASI ---
                depth = 16
                slices = []
                series_dir = os.path.dirname(file_path)
                dcm_files = sorted([f for f in os.listdir(series_dir) if f.lower().endswith('.dcm')])
                
                selected_files = dcm_files[:depth]
                
                for fname in selected_files:
                    try:
                        dcm = pydicom.dcmread(os.path.join(series_dir, fname))
                        img = dcm.pixel_array.astype(np.float32)
                        
                        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                        
                        img_resized = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
                        tensor = torch.from_numpy(img_resized).float() / 255.0
                        slices.append(tensor)
                    except Exception:
                        continue

                while len(slices) < depth:
                    slices.append(torch.zeros((256, 256), dtype=torch.float32))

                volume = torch.stack(slices).unsqueeze(0) # Shape: (1, 16, 256, 256)
                return volume.unsqueeze(0) # DataLoader olmadığı için batch boyutu ekle: (1, 1, 16, 256, 256)
            
            elif self.modality == 'BT':
                # Test.py'nin mantığı: DICOM windowing ve 3 kanala çevirme
                ds = pydicom.dcmread(file_path, force=True)
                arr = _window_image(ds)
                pil_image = Image.fromarray(arr).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                tensor = transform(pil_image)
                return tensor.unsqueeze(0)
            
            return None
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")


class MultiAnalysisWorker(QThread):
    file_progress = pyqtSignal(int, str, list)
    file_error = pyqtSignal(int, str)
    all_finished = pyqtSignal()
    
    def __init__(self, models, device, file_paths, label_names, modality):
        super().__init__()
        self.models = models
        self.device = device
        self.file_paths = file_paths
        self.label_names = label_names
        self.modality = modality

    def run(self):
        for i, file_path in enumerate(self.file_paths):
            try:
                image_tensor = self.preprocess_image(file_path)
                if image_tensor is None:
                    raise Exception("Görüntü işlenemedi.")
                image_tensor_gpu = image_tensor.to(self.device)
                
                if self.modality == 'MR':
                    predictions = []
                    with torch.no_grad():
                        for model in self.models:
                            output = model(image_tensor_gpu)
                            pred_probs = torch.softmax(output, dim=1)
                            predictions.append(pred_probs.cpu().numpy()[0])
                    avg_prediction = np.mean(predictions, axis=0)
                    predicted_class_idx = np.argmax(avg_prediction)
                    all_probabilities = (avg_prediction * 100).tolist()
                elif self.modality == 'BT':
                    outputs = []
                    with torch.no_grad():
                        for model in self.models:
                            output = model(image_tensor_gpu)
                            outputs.append(output.cpu().numpy()[0][0])
                    avg_prob_stroke = np.mean(outputs)
                    all_probabilities_raw = [1 - avg_prob_stroke, avg_prob_stroke]
                    predicted_class_idx = np.argmax(all_probabilities_raw)
                    all_probabilities = [p * 100 for p in all_probabilities_raw]
                predicted_label = self.label_names[predicted_class_idx]
                self.file_progress.emit(i, predicted_label, all_probabilities)
            except Exception as e:
                self.file_error.emit(i, str(e))
        self.all_finished.emit()

    def preprocess_image(self, file_path):
        # AnalysisWorker'daki ile %100 aynı metodu çağır
        return AnalysisWorker.preprocess_image(self, file_path)