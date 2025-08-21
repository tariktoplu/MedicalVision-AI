# workers.py

import torch
import pydicom
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float, list)
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
            self.progress.emit("Görüntü yükleniyor ve işleniyor...")
            image_tensor = self.preprocess_image(self.file_path)
            image_tensor_gpu = image_tensor.to(self.device)
            
            self.progress.emit("Model analizi yapılıyor...")
            predictions = []
            
            with torch.no_grad():
                for i, model in enumerate(self.models):
                    self.progress.emit(f"Model {i+1}/{len(self.models)} çalışıyor...")
                    model.eval()
                    output = model(image_tensor_gpu)
                    pred_probs = torch.softmax(output, dim=1)
                    predictions.append(pred_probs.cpu().numpy()[0])
            
            avg_prediction = np.mean(predictions, axis=0)
            predicted_class = np.argmax(avg_prediction)
            confidence = avg_prediction[predicted_class] * 100
            probabilities = avg_prediction * 100
            
            self.finished.emit(self.label_names[predicted_class], confidence, probabilities.tolist())
            
        except Exception as e:
            self.error.emit(f"Analiz sırasında hata: {str(e)}")
    
    def preprocess_image(self, file_path):
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array.astype(np.float32)
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            # Normalizasyon (Z-score)
            mean = np.mean(image_array)
            std = np.std(image_array)
            image_array = (image_array - mean) / (std + 1e-6)
            
            if self.modality == 'MR':
                # MedNet için ön işleme: (256, 256) boyutunda 16 dilimlik 3D tensör
                image_array = cv2.resize(image_array, (256, 256))
                depth = 16
                slices = [torch.tensor(image_array, dtype=torch.float32) for _ in range(depth)]
                tensor = torch.stack(slices).unsqueeze(0) # Shape: (1, 16, 256, 256)
                return tensor.unsqueeze(0) # Final Shape: (1, 1, 16, 256, 256)
            
            elif self.modality == 'BT':
                # ConvNeXt için ön işleme: (224, 224) boyutunda 1 kanallı 2D tensör
                image_array = cv2.resize(image_array, (224, 224))
                tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0) # Shape: (1, 224, 224)
                return tensor.unsqueeze(0) # Final Shape: (1, 1, 224, 224)
            
            else:
                raise ValueError("Geçersiz modalite.")
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")

class MultiAnalysisWorker(QThread):
    file_progress = pyqtSignal(int, str, float, list)
    file_error = pyqtSignal(int, str)
    all_finished = pyqtSignal()
    
    def __init__(self, models, device, file_paths, label_names, modality):
        super().__init__()
        self.models = models
        self.device = device
        self.file_paths = file_paths
        self.label_names = label_names
        self.modality = modality

    def preprocess_image(self, file_path):
        # AnalysisWorker'daki preprocess_image metodunun aynısını kullanıyoruz
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array.astype(np.float32)
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            mean = np.mean(image_array)
            std = np.std(image_array)
            image_array = (image_array - mean) / (std + 1e-6)
            
            if self.modality == 'MR':
                image_array = cv2.resize(image_array, (256, 256))
                depth = 16
                slices = [torch.tensor(image_array, dtype=torch.float32) for _ in range(depth)]
                tensor = torch.stack(slices).unsqueeze(0)
                return tensor.unsqueeze(0)
            
            elif self.modality == 'BT':
                image_array = cv2.resize(image_array, (224, 224))
                tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
                return tensor.unsqueeze(0)
            
            else:
                raise ValueError("Geçersiz modalite.")
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")

    def run(self):
        for i, file_path in enumerate(self.file_paths):
            try:
                image_tensor = self.preprocess_image(file_path)
                image_tensor_gpu = image_tensor.to(self.device)
                
                predictions = []
                with torch.no_grad():
                    for model in self.models:
                        model.eval()
                        output = model(image_tensor_gpu)
                        pred_probs = torch.softmax(output, dim=1)
                        predictions.append(pred_probs.cpu().numpy()[0])
                
                avg_prediction = np.mean(predictions, axis=0)
                predicted_class = np.argmax(avg_prediction)
                confidence = avg_prediction[predicted_class] * 100
                probabilities = avg_prediction * 100
                
                self.file_progress.emit(i, self.label_names[predicted_class], confidence, probabilities.tolist())
                
            except Exception as e:
                self.file_error.emit(i, str(e))
        
        self.all_finished.emit()