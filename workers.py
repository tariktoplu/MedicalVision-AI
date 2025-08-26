# workers.py

import torch
import pydicom
import numpy as np
import cv2 # Görüntü okuma ve yeniden boyutlandırma için
from PyQt5.QtCore import QThread, pyqtSignal

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
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array.astype(np.float32)
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            if self.modality == 'MR':
                # --- MR İÇİN ORİJİNAL, KANITLANMIŞ VE ÇALIŞAN KODA GERİ DÖNÜLDÜ ---
                mean = np.mean(image_array)
                std = np.std(image_array)
                # Standart sapmanın sıfır olma ihtimaline karşı kontrol
                if std > 0:
                    image_array_normalized = (image_array - mean) / std
                else:
                    image_array_normalized = image_array - mean
                
                image_array_resized = cv2.resize(image_array_normalized, (256, 256))
                
                depth = 16
                slices = [torch.tensor(image_array_resized, dtype=torch.float32) for _ in range(depth)]
                tensor = torch.stack(slices).unsqueeze(0) # Shape: (1, 16, 256, 256)
                
                # MedNet modeli (1, 1, 16, 256, 256) beklediği için kanal boyutu ekle
                return tensor.unsqueeze(0) 
            
            elif self.modality == 'BT':
                # --- BT İÇİN DOĞRU YÖNTEM KORUNDU ---
                if image_array.dtype != np.uint8:
                     array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-6)
                     image_array = (array_normalized * 255).astype(np.uint8)
                
                image_array_resized = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(image_array_resized).float().div(255)
                tensor = tensor.sub(0.5).div(0.5)
                return tensor.unsqueeze(0).unsqueeze(0)
            
            else:
                raise ValueError("Geçersiz modalite.")
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
        # AnalysisWorker'daki ile BİREBİR AYNI metodu kullan
        return AnalysisWorker.preprocess_image(self, file_path)