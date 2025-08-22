# workers.py

import torch
import pydicom
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class AnalysisWorker(QThread):
    # ... (sinyaller aynı)
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
            self.progress.emit("Görüntü işleniyor...")
            image_tensor = self.preprocess_image(self.file_path)
            image_tensor_gpu = image_tensor.to(self.device)
            
            self.progress.emit("Model analizi yapılıyor...")
            
            if self.modality == 'MR':
                # MR için çok sınıflı tahmin
                predictions = []
                with torch.no_grad():
                    for model in self.models:
                        output = model(image_tensor_gpu)
                        pred_probs = torch.softmax(output, dim=1)
                        predictions.append(pred_probs.cpu().numpy()[0])
                
                avg_prediction = np.mean(predictions, axis=0)
                predicted_class_idx = np.argmax(avg_prediction)
                confidence = avg_prediction[predicted_class_idx] * 100
                all_probabilities = (avg_prediction * 100).tolist()
                
            elif self.modality == 'BT':
                # BT için ikili sınıflandırma tahmini
                outputs = []
                with torch.no_grad():
                    for model in self.models:
                        # Model çıktısı tek bir değer (0-1 arası)
                        output = model(image_tensor_gpu)
                        outputs.append(output.cpu().numpy()[0][0])
                
                # Olasılıkların ortalamasını al
                avg_prob_stroke = np.mean(outputs)
                
                if avg_prob_stroke > 0.5:
                    predicted_class_idx = 1 # İnme
                    confidence = avg_prob_stroke * 100
                else:
                    predicted_class_idx = 0 # Sağlıklı
                    confidence = (1 - avg_prob_stroke) * 100
                
                # İki sınıf için de olasılıkları listele
                all_probabilities = [(1 - avg_prob_stroke) * 100, avg_prob_stroke * 100]

            predicted_label = self.label_names[predicted_class_idx]
            self.finished.emit(predicted_label, confidence, all_probabilities)
            
        except Exception as e:
            self.error.emit(f"Analiz hatası: {str(e)}")
    
    # ... (preprocess_image metodu aynı)
    def preprocess_image(self, file_path):
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

# MultiAnalysisWorker sınıfı da benzer şekilde güncellenmeli
class MultiAnalysisWorker(QThread):
    # ... (sinyaller aynı)
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
                    confidence = avg_prediction[predicted_class_idx] * 100
                    all_probabilities = (avg_prediction * 100).tolist()
                
                elif self.modality == 'BT':
                    outputs = []
                    with torch.no_grad():
                        for model in self.models:
                            output = model(image_tensor_gpu)
                            outputs.append(output.cpu().numpy()[0][0])
                    avg_prob_stroke = np.mean(outputs)
                    if avg_prob_stroke > 0.5:
                        predicted_class_idx = 1
                        confidence = avg_prob_stroke * 100
                    else:
                        predicted_class_idx = 0
                        confidence = (1 - avg_prob_stroke) * 100
                    all_probabilities = [(1 - avg_prob_stroke) * 100, avg_prob_stroke * 100]

                predicted_label = self.label_names[predicted_class_idx]
                self.file_progress.emit(i, predicted_label, confidence, all_probabilities)
                
            except Exception as e:
                self.file_error.emit(i, str(e))
        
        self.all_finished.emit()

    def preprocess_image(self, file_path):
        # AnalysisWorker'daki ile aynı
        # ... (Yukarıdaki preprocess_image kodunu buraya kopyalayabilirsiniz)
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