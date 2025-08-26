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
                all_probabilities_raw = [1 - avg_prob_stroke, avg_prob_stroke]
                predicted_class_idx = np.argmax(all_probabilities_raw)
                all_probabilities = [p * 100 for p in all_probabilities_raw]

            predicted_label = self.label_names[predicted_class_idx]
            self.finished.emit(predicted_label, all_probabilities)
            
        except Exception as e:
            self.error.emit(f"Analiz hatası: {str(e)}")
    
    # --- BU METOT, EĞİTİM ORTAMIYLA %100 UYUMLU OLACAK ŞEKİLDE YENİDEN YAZILDI ---
    def preprocess_image(self, file_path):
        try:
            # 1. Görüntüyü bir numpy dizisi olarak oku
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array
            else:
                # PIL, standart formatları okumak için en güvenilir yoldur
                pil_img = Image.open(file_path)
                image_array = np.array(pil_img)

            # 2. Görüntüyü 8-bit grayscale PIL Image nesnesine dönüştür
            # Bu adım, tüm girdi tiplerini standart bir formata getirir.
            # DICOM'dan gelen yüksek bit derinlikli veriyi 0-255 arasına haritalar.
            if image_array.dtype != np.uint8:
                 array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-6)
                 image_array = (array_normalized * 255).astype(np.uint8)

            pil_image = Image.fromarray(image_array).convert("L")
            
            # 3. Modaliteye göre, EĞİTİMDEKİYLE BİREBİR AYNI transform'u uygula
            if self.modality == 'MR':
                # mednet.py'deki mantık: 256x256, 0-1 arası float tensör
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(), # Değerleri [0, 1] arasına getirir
                ])
                tensor_slice = transform(pil_image)
                
                depth = 16
                slices = [tensor_slice for _ in range(depth)]
                tensor = torch.stack(slices, dim=1)
                return tensor.unsqueeze(0)
            
            elif self.modality == 'BT':
                # ConvNext_Tiny.py'deki transform'un AYNISI
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(), # Değerleri [0, 1] arasına getirir
                    transforms.Normalize([0.5], [0.5]) # Değerleri [-1, 1] arasına getirir
                ])
                tensor = transform(pil_image)
                return tensor.unsqueeze(0)
            
            else:
                raise ValueError("Geçersiz modalite.")
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {str(e)}")


class MultiAnalysisWorker(QThread):
    # Bu sınıfın da preprocess_image metodunu güncelleyelim.
    # En kolayı, yukarıdaki AnalysisWorker.preprocess_image metodunu kopyalamaktır.
    
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
        # ... (run metodu önceki doğru haliyle aynı, değişiklik yok)
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
        # AnalysisWorker'daki ile BİREBİR AYNI metodu kullanıyoruz
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array
            else:
                pil_img = Image.open(file_path)
                image_array = np.array(pil_img)
            if image_array.dtype != np.uint8:
                 array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-6)
                 image_array = (array_normalized * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array).convert("L")
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