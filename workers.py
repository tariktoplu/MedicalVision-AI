# workers.py

import os
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import torchvision.transforms as T
from PyQt5.QtCore import QThread, pyqtSignal

# --- LT1.py'den alınan YENİ MR için DICOM işleme fonksiyonu ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def robust_dicom_to_pil_rgb(path: str) -> Image.Image:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    arr = (arr - lo) / denom
    img = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode='L').convert('RGB')

# --- BT için Test(2).py'den alınan DICOM işleme yardımcı fonksiyonları ---
def _window_image(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    data = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    data = data * slope + intercept
    try:
        data = apply_voi_lut(data, ds)
    except Exception:
        center = ds.get('WindowCenter', 40.0)
        width = ds.get('WindowWidth', 80.0)
        if isinstance(center, pydicom.multival.MultiValue): center = float(center[0])
        if isinstance(width, pydicom.multival.MultiValue): width = float(width[0])
        low = center - width / 2.0
        high = center + width / 2.0
        data = np.clip(data, low, high)
    
    if np.ptp(data) > 0:
        data = (data - np.min(data)) / np.ptp(data)
    return (data * 255.0).astype(np.uint8)


class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)
    
    def __init__(self, models, device, file_path, label_names, modality, parent=None):
        super().__init__(parent)
        self.models = models
        self.device = device
        self.file_path = file_path
        self.label_names = label_names
        self.modality = modality
        
    def run(self):
        try:
            if self.isInterruptionRequested(): return
            self.progress.emit("Görüntü işleniyor...")
            image_tensor = self.preprocess_image(self.file_path)
            image_tensor_gpu = image_tensor.to(self.device)
            if self.isInterruptionRequested(): return
            self.progress.emit("Model analizi yapılıyor...")
            
            if self.modality == 'MR':
                logits_list = []
                with torch.no_grad():
                    for model in self.models:
                        output = model(image_tensor_gpu)
                        logits_list.append(output)
                
                avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                probabilities = torch.sigmoid(avg_logits).cpu().numpy()[0]
                
                ML_THRESH = 0.5
                preds_binary = (probabilities >= ML_THRESH).astype(int)
                
                if np.sum(preds_binary) == 0:
                    top_prediction_idx = np.argmax(probabilities)
                    preds_binary = np.zeros_like(preds_binary)
                    preds_binary[top_prediction_idx] = 1
                
                predicted_labels = [self.label_names[i] for i, val in enumerate(preds_binary) if val == 1]
                predicted_label_str = ", ".join(predicted_labels)
                all_probabilities_percent = (probabilities * 100).tolist()
                self.finished.emit(predicted_label_str, all_probabilities_percent)
            
            elif self.modality == 'BT':
                outputs = []
                with torch.no_grad():
                    for model in self.models:
                        if self.isInterruptionRequested(): return
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
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if self.modality == 'MR':
                pil_image = None
                if file_extension == '.dcm':
                    pil_image = robust_dicom_to_pil_rgb(file_path)
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    pil_image = Image.open(file_path).convert('RGB')
                else:
                    raise ValueError(f"Desteklenmeyen format: {file_extension}")

                transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
                tensor = transform(pil_image)
                return tensor.unsqueeze(0)
            
            elif self.modality == 'BT':
                pil_image_rgb = None
                if file_extension == '.dcm':
                    ds = pydicom.dcmread(file_path, force=True)
                    arr = _window_image(ds)
                    pil_image_rgb = Image.fromarray(arr).convert('RGB')
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    pil_image_rgb = Image.open(file_path).convert('RGB')
                else:
                    raise ValueError(f"Desteklenmeyen format: {file_extension}")
                
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                tensor = transform(pil_image_rgb)
                return tensor.unsqueeze(0)
            
            return None
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")


# --- YENİ EKLENDİ: Eksik olan MultiAnalysisWorker sınıfı ---
class MultiAnalysisWorker(QThread):
    file_progress = pyqtSignal(int, str, list)
    file_error = pyqtSignal(int, str)
    all_finished = pyqtSignal()
    
    def __init__(self, models, device, file_paths, label_names, modality, parent=None):
        super().__init__(parent)
        self.models = models
        self.device = device
        self.file_paths = file_paths
        self.label_names = label_names
        self.modality = modality

    def run(self):
        for i, file_path in enumerate(self.file_paths):
            if self.isInterruptionRequested():
                print(f"{self.modality} Analizi kullanıcı tarafından iptal edildi.")
                return
            try:
                # AnalysisWorker'daki preprocess metodunu çağırarak tutarlılığı sağla
                temp_worker = AnalysisWorker(self.models, self.device, file_path, self.label_names, self.modality)
                image_tensor = temp_worker.preprocess_image(file_path)
                image_tensor_gpu = image_tensor.to(self.device)
                
                if self.modality == 'MR':
                    # Bu worker'ın MR versiyonu MultiAnalysisPageMR'da, bu bloğa girilmemeli
                    # Ancak bir güvenlik önlemi olarak buraya da ekleyebiliriz
                    pass
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
        
        if not self.isInterruptionRequested():
            self.all_finished.emit()