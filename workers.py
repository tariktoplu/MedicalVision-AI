# medical_analyzer_project/workers.py

import torch
import pydicom
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class AnalysisWorker(QThread):
    """Analiz işlemi için worker thread"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float, list)
    error = pyqtSignal(str)
    
    def __init__(self, models, device, file_path, label_names):
        super().__init__()
        self.models = models
        self.device = device
        self.file_path = file_path
        self.label_names = label_names
        
    def run(self):
        try:
            self.progress.emit("Görüntü yükleniyor...")
            image_tensor = self.preprocess_image(self.file_path)
            
            self.progress.emit("Model analizi yapılıyor...")
            predictions = []
            
            with torch.no_grad():
                for i, model in enumerate(self.models):
                    self.progress.emit(f"Model {i+1}/{len(self.models)} çalışıyor...")
                    model.eval()
                    output = model(image_tensor.unsqueeze(0).to(self.device))
                    pred_probs = torch.softmax(output, dim=1)
                    predictions.append(pred_probs.cpu().numpy()[0])
            
            # Ensemble tahmini
            avg_prediction = np.mean(predictions, axis=0)
            predicted_class = np.argmax(avg_prediction)
            confidence = avg_prediction[predicted_class] * 100
            probabilities = avg_prediction * 100
            
            self.finished.emit(self.label_names[predicted_class], confidence, probabilities.tolist())
            
        except Exception as e:
            self.error.emit(str(e))
    
    def preprocess_image(self, file_path):
        """Görüntüyü model için hazırla"""
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array.astype(np.float32)
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            # Normalizasyon
            image_array = (image_array - image_array.mean()) / (image_array.std() + 1e-6)
            image_array = cv2.resize(image_array, (256, 256))
            
            # 3D tensor oluştur
            depth = 16
            slices = []
            for _ in range(depth):
                slices.append(torch.tensor(image_array))
            
            tensor = torch.stack(slices).unsqueeze(0)
            return tensor
            
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")

class MultiAnalysisWorker(QThread):
    """Çoklu analiz için worker thread"""
    file_progress = pyqtSignal(int, str, float, list)  # index, prediction, confidence, probabilities
    file_error = pyqtSignal(int, str)  # index, error
    all_finished = pyqtSignal()
    
    def __init__(self, models, device, file_paths, label_names):
        super().__init__()
        self.models = models
        self.device = device
        self.file_paths = file_paths
        self.label_names = label_names
        
    def run(self):
        for i, file_path in enumerate(self.file_paths):
            try:
                image_tensor = self.preprocess_image(file_path)
                
                predictions = []
                with torch.no_grad():
                    for model in self.models:
                        model.eval()
                        output = model(image_tensor.unsqueeze(0).to(self.device))
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
    
    def preprocess_image(self, file_path):
        """Görüntü ön işleme"""
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array.astype(np.float32)
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            image_array = (image_array - image_array.mean()) / (image_array.std() + 1e-6)
            image_array = cv2.resize(image_array, (256, 256))
            
            depth = 16
            slices = []
            for _ in range(depth):
                slices.append(torch.tensor(image_array))
            
            tensor = torch.stack(slices).unsqueeze(0)
            return tensor
            
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")