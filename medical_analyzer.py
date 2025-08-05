import sys
import os
import json
import torch
import pydicom
import numpy as np
from PIL import Image
import cv2
import threading
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Model sınıfları (train.py'den alındı)
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = x.mean(dim=[2, 3, 4])
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y

class MR3DCNN_LSTM_Attention(nn.Module):
    def __init__(self, hidden_size=64, lstm_layers=1):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.se1 = SEBlock(8)
        
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.se2 = SEBlock(16)
        
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(input_size=16 * 64 * 64, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.se1(x)
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.se2(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

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

class StartPage(QWidget):
    """Başlangıç sayfası"""
    modality_selected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Başlık
        title = QLabel(" Medikal Görüntü Analiz Sistemi")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin: 20px;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-radius: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Modalite seçim başlığı
        subtitle = QLabel("Görüntü Modalitesi Seçiniz:")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; font-weight: bold; color: #34495e; margin: 10px;")
        layout.addWidget(subtitle)
        
        # Buton container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(50)
        
        # BT Butonu
        bt_btn = QPushButton("BT")
        bt_btn.setFixedSize(200, 100)
        bt_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #5dade2, stop:1 #3498db);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #2980b9, stop:1 #1b4f72);
            }
        """)
        bt_btn.clicked.connect(lambda: self.modality_selected.emit("BT"))
        button_layout.addWidget(bt_btn)
        
        # MR Butonu
        mr_btn = QPushButton("MR")
        mr_btn.setFixedSize(200, 100)
        mr_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ec7063, stop:1 #e74c3c);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #c0392b, stop:1 #922b21);
            }
        """)
        mr_btn.clicked.connect(lambda: self.modality_selected.emit("MR"))
        button_layout.addWidget(mr_btn)
        
        layout.addWidget(button_container)
        layout.addStretch()
        
        self.setLayout(layout)

class AnalysisModePage(QWidget):
    """Analiz modu seçim sayfası"""
    mode_selected = pyqtSignal(str, str)  # modality, mode
    back_clicked = pyqtSignal()
    
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Geri butonu
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(100, 40)
        back_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                background: #95a5a6;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover { background: #7f8c8d; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        
        back_layout = QHBoxLayout()
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        
        # Başlık
        title = QLabel(f"{self.modality} Analiz Modu")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 20px;
                padding: 15px;
                background: #ecf0f1;
                border-radius: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Mod butonları
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(50)
        
        # Tekli Analiz
        single_btn = QPushButton(" Tekli Analiz")
        single_btn.setFixedSize(200, 100)
        single_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #27ae60, stop:1 #229954);
                color: white;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #58d68d, stop:1 #27ae60);
            }
        """)
        single_btn.clicked.connect(lambda: self.mode_selected.emit(self.modality, "single"))
        button_layout.addWidget(single_btn)
        
        # Çoklu Analiz
        multi_btn = QPushButton(" Çoklu Analiz")
        multi_btn.setFixedSize(200, 100)
        multi_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f39c12, stop:1 #e67e22);
                color: white;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f8c471, stop:1 #f39c12);
            }
        """)
        multi_btn.clicked.connect(lambda: self.mode_selected.emit(self.modality, "multi"))
        button_layout.addWidget(multi_btn)
        
        layout.addWidget(button_container)
        layout.addStretch()
        
        self.setLayout(layout)

class SingleAnalysisPage(QWidget):
    """Tekli analiz sayfası"""
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality
        self.models = models
        self.device = device
        self.label_names = label_names
        self.current_file = None
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Üst bar
        top_bar = QHBoxLayout()
        
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(80, 35)
        back_btn.setStyleSheet("""
            QPushButton {
                background: #95a5a6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 12px;
            }
            QPushButton:hover { background: #7f8c8d; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        
        title = QLabel(f"{self.modality} - Tekli Analiz")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        
        main_layout.addLayout(top_bar)
        
        # Ana içerik
        content_layout = QHBoxLayout()
        
        # Sol panel - Dosya yükleme ve önizleme
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        left_panel.setFixedWidth(400)
        
        left_layout = QVBoxLayout(left_panel)
        
        # Dosya yükleme
        upload_section = QLabel(" Dosya Yükleme")
        upload_section.setAlignment(Qt.AlignCenter)
        upload_section.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(upload_section)
        
        self.upload_btn = QPushButton("Dosya Seç")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover { background: #2980b9; }
        """)
        self.upload_btn.clicked.connect(self.upload_file)
        left_layout.addWidget(self.upload_btn)
        
        # Dosya bilgisi
        self.file_info = QLabel("Dosya seçilmedi")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setStyleSheet("margin: 10px; color: #7f8c8d;")
        left_layout.addWidget(self.file_info)
        
        # Önizleme
        preview_label = QLabel(" Önizleme")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(preview_label)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet("border: 2px dashed #bdc3c7; border-radius: 8px;")
        self.preview_label.setText("Görüntü yükleyin")
        left_layout.addWidget(self.preview_label)
        
        content_layout.addWidget(left_panel)
        
        # Sağ panel - Sonuçlar
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        
        right_layout = QVBoxLayout(right_panel)
        
        result_title = QLabel(" Analiz Sonuçları")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        right_layout.addWidget(result_title)
        
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_area.setStyleSheet("border: none;")
        
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)
        self.result_area.setWidget(self.result_widget)
        
        # Başlangıç mesajı
        initial_msg = QLabel("Analiz için dosya yükleyin")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setStyleSheet("color: #95a5a6; font-size: 14px; margin: 50px;")
        self.result_layout.addWidget(initial_msg)
        
        right_layout.addWidget(self.result_area)
        
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
    
    def upload_file(self):
        """Dosya yükleme"""
        file_types = "Tüm Desteklenen (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Görüntü (*.png *.jpg *.jpeg *.bmp)"
        file_path, _ = QFileDialog.getOpenFileName(self, f"{self.modality} Dosyası Seç", "", file_types)
        
        if file_path:
            self.current_file = file_path
            self.show_file_info(file_path)
            self.show_preview(file_path)
            self.analyze_file(file_path)
    
    def show_file_info(self, file_path):
        """Dosya bilgilerini göster"""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].upper()
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        info_text = f"{file_name}\n{file_ext} • {file_size:.1f} KB"
        self.file_info.setText(info_text)
        self.file_info.setStyleSheet("margin: 10px; color: #2c3e50; font-weight: bold;")
    
    def show_preview(self, file_path):
        """Görüntü önizlemesi"""
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Normalizasyon ve boyutlandırma
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            # PIL ve QPixmap'e çevir
            pil_image = Image.fromarray(image_array)
            pil_image.thumbnail((280, 280), Image.Resampling.LANCZOS)
            
            # QPixmap'e çevir
            qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 8px;")
            
        except Exception as e:
            self.preview_label.setText(f"Önizleme hatası:\n{str(e)}")
            self.preview_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 8px; color: red;")
    
    def analyze_file(self, file_path):
        """Dosya analizi"""
        # Sonuç alanını temizle
        for i in reversed(range(self.result_layout.count())):
            self.result_layout.itemAt(i).widget().deleteLater()
        
        # Progress göstergesi
        self.progress_label = QLabel(" Analiz başlatılıyor...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; color: #3498db; margin: 20px;")
        self.result_layout.addWidget(self.progress_label)
        
        # Worker thread başlat
        self.worker = AnalysisWorker(self.models, self.device, file_path, self.label_names)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, message):
        """Progress güncelleme"""
        self.progress_label.setText(f" {message}")
    
    def show_results(self, prediction, confidence, probabilities):
        """Sonuçları göster"""
        # Progress label'ı kaldır
        self.progress_label.deleteLater()
        
        # Başarı mesajı
        success_label = QLabel(" Analiz Tamamlandı!")
        success_label.setAlignment(Qt.AlignCenter)
        success_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60; margin: 10px;")
        self.result_layout.addWidget(success_label)
        
        # Ana sonuç
        result_frame = QFrame()
        result_frame.setStyleSheet("QFrame { background: #e8f5e8; border-radius: 10px; padding: 15px; }")
        result_layout = QVBoxLayout(result_frame)
        
        pred_label = QLabel(" Tahmin:")
        pred_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        result_layout.addWidget(pred_label)
        
        pred_value = QLabel(prediction)
        pred_value.setStyleSheet("font-size: 20px; font-weight: bold; color: #e74c3c; margin: 5px;")
        result_layout.addWidget(pred_value)
        
        conf_label = QLabel(f" Güven Oranı: %{confidence:.1f}")
        conf_label.setStyleSheet("font-size: 14px; color: #2c3e50; margin: 5px;")
        result_layout.addWidget(conf_label)
        
        self.result_layout.addWidget(result_frame)
        
        # Tüm sınıf olasılıkları
        prob_frame = QFrame()
        prob_frame.setStyleSheet("QFrame { background: #f8f9fa; border-radius: 10px; padding: 15px; }")
        prob_layout = QVBoxLayout(prob_frame)
        
        prob_title = QLabel(" Tüm Sınıf Olasılıkları:")
        prob_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        prob_layout.addWidget(prob_title)
        
        for i, (label, prob) in enumerate(zip(self.label_names, probabilities)):
            prob_item = QLabel(f"• {label}: %{prob:.1f}")
            color = "#e74c3c" if label == prediction else "#7f8c8d"
            prob_item.setStyleSheet(f"font-size: 12px; color: {color}; margin: 2px;")
            prob_layout.addWidget(prob_item)
        
        self.result_layout.addWidget(prob_frame)
        self.result_layout.addStretch()
    
    def show_error(self, error_message):
        """Hata gösterimi"""
        self.progress_label.deleteLater()
        
        error_label = QLabel(" Analiz Hatası")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c; margin: 20px;")
        self.result_layout.addWidget(error_label)
        
        error_detail = QLabel(error_message)
        error_detail.setAlignment(Qt.AlignCenter)
        error_detail.setWordWrap(True)
        error_detail.setStyleSheet("font-size: 12px; color: #e74c3c; margin: 10px;")
        self.result_layout.addWidget(error_detail)

class MultiAnalysisPage(QWidget):
    """Çoklu analiz sayfası"""
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality
        self.models = models
        self.device = device
        self.label_names = label_names
        self.file_paths = []
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Üst bar
        top_bar = QHBoxLayout()
        
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(80, 35)
        back_btn.setStyleSheet("""
            QPushButton {
                background: #95a5a6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 12px;
            }
            QPushButton:hover { background: #7f8c8d; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        
        title = QLabel(f"{self.modality} - Çoklu Analiz")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        
        main_layout.addLayout(top_bar)
        
        # Ana içerik
        content_layout = QHBoxLayout()
        
        # Sol panel - Dosya yükleme
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        left_panel.setFixedWidth(400)
        
        left_layout = QVBoxLayout(left_panel)
        
        # Dosya yükleme
        upload_section = QLabel(" Dosya Yükleme")
        upload_section.setAlignment(Qt.AlignCenter)
        upload_section.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(upload_section)
        
        self.upload_btn = QPushButton("Dosya Seç")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover { background: #2980b9; }
        """)
        self.upload_btn.clicked.connect(self.upload_files)
        left_layout.addWidget(self.upload_btn)
        
        # Dosya bilgisi
        self.file_info = QLabel("Henüz dosya yüklenmedi")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setStyleSheet("margin: 10px; color: #7f8c8d;")
        left_layout.addWidget(self.file_info)
        
        # Dosya tablosu
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Dosya Adı', 'Format', 'Durum', 'Tahmin', 'Güven (%)'])
        
        # Kolon genişlikleri
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        # Tablo stili
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #bdc3c7;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #3498db;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """)
        left_layout.addWidget(self.progress_bar)
        
        content_layout.addWidget(left_panel)
        
        # Sağ panel - Sonuçlar
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        
        right_layout = QVBoxLayout(right_panel)
        
        result_title = QLabel(" Analiz Sonuçları")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        right_layout.addWidget(result_title)
        
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_area.setStyleSheet("border: none;")
        
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)
        self.result_area.setWidget(self.result_widget)
        
        # Başlangıç mesajı
        initial_msg = QLabel("Analiz için dosyaları yükleyin")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setStyleSheet("color: #95a5a6; font-size: 14px; margin: 50px;")
        self.result_layout.addWidget(initial_msg)
        
        right_layout.addWidget(self.result_area)
        
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
    
    def upload_files(self):
        """Çoklu dosya yükleme"""
        file_types = "Tüm Desteklenen (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Görüntü (*.png *.jpg *.jpeg *.bmp)"
        file_paths, _ = QFileDialog.getOpenFileNames(self, f"{self.modality} Dosyalarını Seç", "", file_types)
        
        if file_paths:
            self.file_paths.extend(file_paths)
            self.populate_table()
            self.start_analysis()
    
    def populate_table(self):
        """Tabloyu doldur"""
        self.table.setRowCount(len(self.file_paths))
        
        for i, file_path in enumerate(self.file_paths):
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].upper()
            
            self.table.setItem(i, 0, QTableWidgetItem(file_name))
            self.table.setItem(i, 1, QTableWidgetItem(file_ext))
            self.table.setItem(i, 2, QTableWidgetItem("Bekliyor"))
            self.table.setItem(i, 3, QTableWidgetItem("-"))
            self.table.setItem(i, 4, QTableWidgetItem("-"))
            
            # Durum sütunu stilini ayarla
            status_item = self.table.item(i, 2)
            status_item.setBackground(QColor("#f39c12"))
            status_item.setForeground(QColor("white"))
    
    def start_analysis(self):
        """Analizi başlat"""
        if not self.file_paths:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        
        # Worker thread başlat
        self.worker = MultiAnalysisWorker(self.models, self.device, self.file_paths, self.label_names)
        self.worker.file_progress.connect(self.update_file_result)
        self.worker.file_error.connect(self.update_file_error)
        self.worker.all_finished.connect(self.analysis_finished)
        self.worker.start()
    
    def update_file_result(self, index, prediction, confidence, probabilities):
        """Dosya sonucunu güncelle"""
        self.table.setItem(index, 2, QTableWidgetItem("Tamamlandı"))
        self.table.setItem(index, 3, QTableWidgetItem(prediction))
        self.table.setItem(index, 4, QTableWidgetItem(f"{confidence:.1f}"))
        
        # Durum sütunu stilini güncelle
        status_item = self.table.item(index, 2)
        status_item.setBackground(QColor("#27ae60"))
        status_item.setForeground(QColor("white"))
        
        # Tahmin stilini ayarla
        pred_item = self.table.item(index, 3)
        pred_item.setForeground(QColor("#e74c3c"))
        pred_item.setFont(QFont("Arial", -1, QFont.Bold))
        
        # Progress bar güncelle
        self.progress_bar.setValue(self.progress_bar.value() + 1)
    
    def update_file_error(self, index, error_message):
        """Dosya hatasını güncelle"""
        self.table.setItem(index, 2, QTableWidgetItem("Hata"))
        self.table.setItem(index, 3, QTableWidgetItem(error_message[:30] + "..."))
        self.table.setItem(index, 4, QTableWidgetItem("-"))
        
        # Durum sütunu stilini güncelle
        status_item = self.table.item(index, 2)
        status_item.setBackground(QColor("#e74c3c"))
        status_item.setForeground(QColor("white"))
        
        # Progress bar güncelle
        self.progress_bar.setValue(self.progress_bar.value() + 1)
    
    def analysis_finished(self):
        """Analiz tamamlandı"""
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Analiz Tamamlandı", 
                               f"{len(self.file_paths)} dosyanın analizi tamamlandı!")
    
    def clear_files(self):
        """Dosyaları temizle"""
        self.file_paths.clear()
        self.table.setRowCount(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

class MedicalImageAnalyzer(QMainWindow):
    """Ana uygulama sınıfı"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medikal Görüntü Analiz Sistemi")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Modern stil
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
        """)
        
        # Model yükleme
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models()
        self.label_names = ['HiperakutAkut', 'Subakut', 'NormalKronik']
        
        # Stack widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Sayfaları oluştur
        self.start_page = StartPage()
        self.start_page.modality_selected.connect(self.show_mode_page)
        self.stack.addWidget(self.start_page)
        
        # Durum çubuğu
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Hazır - {len(self.models)} model yüklendi - Cihaz: {self.device}")
        
        # İkon ayarla (opsiyonel)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
    
    def load_models(self):
        """5 fold modelini yükle"""
        models = []
        model_dir = "Models"
        
        try:
            for fold in range(5):
                model_path = f"{model_dir}/best_model_fold_{fold}.pt"
                if os.path.exists(model_path):
                    model = MR3DCNN_LSTM_Attention()
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    models.append(model)
                    print(f"Model fold {fold} yüklendi")
                else:
                    print(f"Uyarı: {model_path} bulunamadı")
            
            if not models:
                QMessageBox.critical(None, "Model Hatası", 
                                   "Hiçbir model dosyası bulunamadı!\n"
                                   "results_3DCNN/ klasöründe model dosyalarını kontrol edin.")
            else:
                print(f"Toplam {len(models)} model başarıyla yüklendi")
                
        except Exception as e:
            QMessageBox.critical(None, "Model Yükleme Hatası", 
                               f"Modeller yüklenirken hata oluştu:\n{str(e)}")
        
        return models
    
    def show_mode_page(self, modality):
        """Mod seçim sayfasını göster"""
        mode_page = AnalysisModePage(modality)
        mode_page.mode_selected.connect(self.show_analysis_page)
        mode_page.back_clicked.connect(self.show_start_page)
        
        self.stack.addWidget(mode_page)
        self.stack.setCurrentWidget(mode_page)
    
    def show_analysis_page(self, modality, mode):
        """Analiz sayfasını göster"""
        if mode == "single":
            analysis_page = SingleAnalysisPage(modality, self.models, self.device, self.label_names)
        else:
            analysis_page = MultiAnalysisPage(modality, self.models, self.device, self.label_names)
        
        analysis_page.back_clicked.connect(self.show_start_page)
        
        self.stack.addWidget(analysis_page)
        self.stack.setCurrentWidget(analysis_page)
    
    def show_start_page(self):
        """Başlangıç sayfasını göster"""
        # Eski sayfaları temizle (memory leak önleme)
        while self.stack.count() > 1:
            widget = self.stack.widget(1)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        
        self.stack.setCurrentWidget(self.start_page)

def main():
    """Ana fonksiyon"""
    app = QApplication(sys.argv)
    
    # Uygulama bilgileri
    app.setApplicationName("Medikal Görüntü Analiz Sistemi")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Medical AI Solutions")
    
    # Modern stil
    app.setStyle('Fusion')
    
    # Dark palette (opsiyonel)
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # app.setPalette(palette)
    
    # Ana pencereyi oluştur ve göster
    window = MedicalImageAnalyzer()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()