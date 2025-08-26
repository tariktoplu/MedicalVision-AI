# workers.py

import torch
import pydicom
import numpy as np
# DEĞİŞTİ: cv2 yerine PIL ve torchvision.transforms kullanacağız
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
            # ... (run metodunun geri kalanı aynı, değişiklik yok)
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
                predicted_class_idx = 1 if avg_prob_stroke > 0.5 else 0
                all_probabilities = [(1 - avg_prob_stroke) * 100, avg_prob_stroke * 100]
            predicted_label = self.label_names[predicted_class_idx]
            self.finished.emit(predicted_label, all_probabilities)
        except Exception as e:
            self.error.emit(f"Analiz hatası: {str(e)}")
    
    # --- DEĞİŞTİ: Ön işleme tamamen eğitim kodunu taklit edecek şekilde yeniden yazıldı ---
    def preprocess_image(self, file_path):
        try:
            # Görüntüyü PIL Image nesnesi olarak yükle
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                # DICOM piksel verisini 0-255 aralığında uint8'e çevir
                array = dcm.pixel_array
                array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
                pil_image = Image.fromarray(array.astype(np.uint8)).convert("L") # "L" modu: grayscale
            else:
                pil_image = Image.open(file_path).convert("L") # Grayscale'e çevir
            
            # Modaliteye göre doğru transformu uygula
            if self.modality == 'MR':
                # mednet.py eğitim kodundaki gibi: 256x256, 0-1 arası float tensör
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(), # Değerleri 0-1 arasına getirir
                ])
                tensor_slice = transform(pil_image)
                
                # 3D tensör oluştur
                depth = 16
                slices = [tensor_slice for _ in range(depth)]
                tensor = torch.stack(slices, dim=1) # Shape: (1, 16, 256, 256)
                return tensor.unsqueeze(0) # Final Shape: (1, 1, 16, 256, 256)
            
            elif self.modality == 'BT':
                # ConvNext_Tiny.py eğitim kodundakiyle BİREBİR AYNI transform
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(), # Değerleri 0-1 arasına getirir
                    transforms.Normalize([0.5], [0.5]) # Değerleri -1 ile 1 arasına getirir
                ])
                tensor = transform(pil_image)
                return tensor.unsqueeze(0) # Final Shape: (1, 1, 224, 224)
            
            else:
                raise ValueError("Geçersiz modalite.")
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")

# MultiAnalysisWorker sınıfı da aynı şekilde güncellenmeli
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
        # ... (run metodu aynı, değişiklik yok)
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
                    predicted_class_idx = 1 if avg_prob_stroke > 0.5 else 0
                    all_probabilities = [(1 - avg_prob_stroke) * 100, avg_prob_stroke * 100]
                predicted_label = self.label_names[predicted_class_idx]
                self.file_progress.emit(i, predicted_label, all_probabilities)
            except Exception as e:
                self.file_error.emit(i, str(e))
        self.all_finished.emit()

    def preprocess_image(self, file_path):
        # AnalysisWorker'daki ile BİREBİR AYNI metodu kullanıyoruz
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                array = dcm.pixel_array
                array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
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