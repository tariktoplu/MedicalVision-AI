# workers.py

import torch
import pydicom
import numpy as np
from PIL import Image
import torchvision.transforms as transforms 
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
                
                # Olasılıkları bir liste olarak tutalım: [Sağlıklı Olasılığı, İnme Olasılığı]
                all_probabilities_raw = [1 - avg_prob_stroke, avg_prob_stroke]
                
                # --- YENİ MANTIK: Hangi olasılık daha yüksekse, o sınıfı seç ---
                # np.argmax, en yüksek değerin indeksini döndürür.
                # Eğer Sağlıklı olasılığı (%52) İnme olasılığından (%48) yüksekse 0 döndürür.
                # Eğer İnme olasılığı (%60) Sağlıklı olasılığından (%40) yüksekse 1 döndürür.
                predicted_class_idx = np.argmax(all_probabilities_raw)
                
                # Olasılıkları yüzdeye çevir
                all_probabilities = [p * 100 for p in all_probabilities_raw]

            predicted_label = self.label_names[predicted_class_idx]
            self.finished.emit(predicted_label, all_probabilities)
            
        except Exception as e:
            self.error.emit(f"Analiz hatası: {str(e)}")
    
    def preprocess_image(self, file_path):
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                array = dcm.pixel_array
                array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-6) * 255
                pil_image = Image.fromarray(array.astype(np.uint8)).convert("L")
            else:
                pil_image = Image.open(file_path).convert("L")
            
            if self.modality == 'MR':
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])
                tensor_slice = transform(pil_image)
                depth = 16
                slices = [tensor_slice for _ in range(depth)]
                tensor = torch.stack(slices, dim=1)
                return tensor.unsqueeze(0)
            
            elif self.modality == 'BT':
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
                tensor = transform(pil_image)
                return tensor.unsqueeze(0)
            
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
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                array = dcm.pixel_array
                array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-6) * 255
                pil_image = Image.fromarray(array.astype(np.uint8)).convert("L")
            else:
                pil_image = Image.open(file_path).convert("L")
            
            if self.modality == 'MR':
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])
                tensor_slice = transform(pil_image)
                depth = 16
                slices = [tensor_slice for _ in range(depth)]
                tensor = torch.stack(slices, dim=1)
                return tensor.unsqueeze(0)
            
            elif self.modality == 'BT':
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
                tensor = transform(pil_image)
                return tensor.unsqueeze(0)
            
            else:
                raise ValueError("Geçersiz modalite.")
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")