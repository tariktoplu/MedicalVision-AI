# workers.py

import os
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import torchvision.transforms as T
from PyQt5.QtCore import QThread, pyqtSignal

# ==============================================================================
# --- LT.py ve L.py'den BİREBİR KOPYALANAN YARDIMCI FONKSİYONLAR ---
# ==============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def process_array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """
    LT.py'deki robust_dicom_to_pil_rgb'nin ana mantığını alır ve
    herhangi bir numpy dizisine uygular. Bu, tutarlılığı garanti eder.
    """
    arr = arr.astype(np.float32)
    # HU birimleri varsayımı (slope/intercept) sadece DICOM için geçerli,
    # PNG/JPG için bu adımı atlıyoruz.
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    arr = (arr - lo) / denom
    img = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode='L').convert('RGB')

def _window_image(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """BT için kullanılan orijinal fonksiyon."""
    data = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    data = data * slope + intercept
    try:
        data = apply_voi_lut(data, ds)
    except Exception:
        center = float(ds.get('WindowCenter', 40.0))
        width = float(ds.get('WindowWidth', 80.0))
        if isinstance(center, pydicom.multival.MultiValue): center = float(center[0])
        if isinstance(width, pydicom.multival.MultiValue): width = float(width[0])
        low = center - width / 2.0
        high = center + width / 2.0
        data = np.clip(data, low, high)
    
    if np.ptp(data) > 0:
        data = (data - np.min(data)) / np.ptp(data)
    return (data * 255.0).astype(np.uint8)
# ==============================================================================


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
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if self.modality == 'MR':
                # 1. Görüntüyü formatı ne olursa olsun bir numpy dizisine oku
                if file_extension == '.dcm':
                    ds = pydicom.dcmread(file_path)
                    arr = ds.pixel_array.astype(np.float32)
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    arr = arr * slope + intercept
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    arr = np.array(Image.open(file_path), dtype=np.float32)
                else:
                    raise ValueError(f"Desteklenmeyen format: {file_extension}")

                # 2. LT.py'deki kontrast ayarlama ve RGB'ye çevirme mantığını uygula
                pil_image_rgb = process_array_to_pil_rgb(arr)

                # 3. LT.py'deki transform'un birebir aynısını uygula
                transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
                tensor = transform(pil_image_rgb)
                return tensor.unsqueeze(0)
            
            elif self.modality == 'BT':
                # Bu mantık zaten doğruydu, aynı kalıyor
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
        return AnalysisWorker.preprocess_image(self, file_path)