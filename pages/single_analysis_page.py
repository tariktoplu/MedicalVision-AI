# pages/single_analysis_page.py

import os
import pydicom
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QScrollArea, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDragLeaveEvent, QDropEvent

from workers import AnalysisWorker

class SingleAnalysisPage(QWidget):
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality
        self.models = models
        self.device = device
        self.label_names = label_names
        self.current_file = None
        self.setAcceptDrops(True)
        self.setup_ui()

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()
            self.preview_label.setStyleSheet("border: 3px dashed #3498db; border-radius: 8px;")

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        if not self.current_file:
            self.preview_label.setStyleSheet("border: 2px dashed #bdc3c7; border-radius: 8px;")
        else:
            self.preview_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 8px;")

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            file_path = event.mimeData().urls()[0].toLocalFile()
            supported_formats = ('.dcm', '.png', '.jpg', '.jpeg')
            if file_path.lower().endswith(supported_formats):
                self.process_new_file(file_path)
            else:
                QMessageBox.warning(self, "Desteklenmeyen Dosya", "Lütfen desteklenen bir görüntü formatı seçin.")
                self.dragLeaveEvent(None)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        top_bar = QHBoxLayout()
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(80, 35)
        back_btn.setStyleSheet("QPushButton { background: #95a5a6; color: white; border: none; border-radius: 8px; font-size: 12px; } QPushButton:hover { background: #7f8c8d; }")
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        title = QLabel(f"{self.modality} - Tekli Analiz")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        content_layout = QHBoxLayout()
        self.left_panel = QFrame()
        self.left_panel.setFrameStyle(QFrame.StyledPanel)
        self.left_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        self.left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(self.left_panel)
        upload_section = QLabel(" Dosya Yükleme")
        upload_section.setAlignment(Qt.AlignCenter)
        upload_section.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(upload_section)
        self.upload_btn = QPushButton("Dosya Seç")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("QPushButton { background: #3498db; color: white; border: none; border-radius: 8px; font-size: 14px; } QPushButton:hover { background: #2980b9; }")
        self.upload_btn.clicked.connect(self.upload_file)
        left_layout.addWidget(self.upload_btn)
        self.file_info = QLabel("Dosya seçilmedi")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setStyleSheet("margin: 10px; color: #7f8c8d;")
        left_layout.addWidget(self.file_info)
        preview_title = QLabel(" Önizleme")
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(preview_title)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet("border: 2px dashed #bdc3c7; border-radius: 8px;")
        self.preview_label.setText("Dosyayı Buraya Sürükleyin\nveya 'Dosya Seç' Butonuna Tıklayın")
        self.preview_label.setWordWrap(True)
        left_layout.addWidget(self.preview_label)
        content_layout.addWidget(self.left_panel)
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
        initial_msg = QLabel("Analiz için dosya yükleyin")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setStyleSheet("color: #95a5a6; font-size: 14px; margin: 50px;")
        self.result_layout.addWidget(initial_msg)
        right_layout.addWidget(self.result_area)
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)

    def process_new_file(self, file_path):
        self.current_file = file_path
        self.show_file_info(file_path)
        self.show_preview(file_path)
        self.analyze_file(file_path)

    def upload_file(self):
        file_types = "Tüm Desteklenen Görüntüler (*.dcm *.png *.jpg *.jpeg);;DICOM (*.dcm);;Standart Görüntüler (*.png *.jpg *.jpeg)"
        file_path, _ = QFileDialog.getOpenFileName(self, f"{self.modality} Dosyası Seç", "", file_types)
        if file_path:
            self.process_new_file(file_path)

    def show_file_info(self, file_path):
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].upper()
        file_size = os.path.getsize(file_path) / 1024
        info_text = f"{file_name}\n{file_ext} • {file_size:.1f} KB"
        self.file_info.setText(info_text)
        self.file_info.setStyleSheet("margin: 10px; color: #2c3e50; font-weight: bold;")

    # --- DEĞİŞTİ: BT DICOM'lar için özel pencereleme eklendi ---
    def show_preview(self, file_path):
        try:
            image_array_8bit = None
            if file_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(file_path)
                image = ds.pixel_array.astype(np.int16)

                if self.modality == 'BT':
                    # --- YENİ: BT için radyolojik pencereleme ---
                    intercept = getattr(ds, "RescaleIntercept", 0)
                    slope = getattr(ds, "RescaleSlope", 1)
                    image = image * slope + intercept
                    
                    # Beyin Penceresi: Window Center=40, Window Width=80
                    window_center = 40
                    window_width = 80
                    img_min = window_center - window_width // 2
                    img_max = window_center + window_width // 2
                    image = np.clip(image, img_min, img_max)
                    
                    # Görüntüyü 0-255 arasına normalize et
                    if np.max(image) != np.min(image):
                        image = (image - np.min(image)) / (np.max(image) - np.min(image))
                    image_array_8bit = (image * 255).astype(np.uint8)

                else: # MR için basit min-max normalizasyon
                    if np.max(image) != np.min(image):
                        image = (image - np.min(image)) / (np.max(image) - np.min(image))
                    image_array_8bit = (image * 255).astype(np.uint8)
            
            else: # PNG, JPG, vb. için
                image_array_8bit = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if image_array_8bit is None:
                raise ValueError("Görüntü verisi okunamadı.")
            
            final_pil_image = Image.fromarray(image_array_8bit)
            final_pil_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            qimage = QImage(final_pil_image.tobytes(), final_pil_image.width, final_pil_image.height, final_pil_image.width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 8px;")
            
        except Exception as e:
            self.preview_label.setText(f"Önizleme hatası:\n{str(e)}")
            self.preview_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 8px; color: red;")
            self.current_file = None

    def analyze_file(self, file_path):
        # ... (Bu metot aynı, değişiklik yok)
        self.clear_layout(self.result_layout)
        self.progress_label = QLabel(" Analiz başlatılıyor...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; color: #3498db; margin: 20px;")
        self.result_layout.addWidget(self.progress_label)
        self.worker = AnalysisWorker(self.models, self.device, file_path, self.label_names, self.modality)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, message):
        self.progress_label.setText(f" {message}")

    def show_results(self, prediction, probabilities):
        # ... (Bu metot aynı, değişiklik yok)
        self.clear_layout(self.result_layout)
        success_label = QLabel(" Analiz Tamamlandı!")
        success_label.setAlignment(Qt.AlignCenter)
        success_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60; margin: 10px;")
        self.result_layout.addWidget(success_label)
        result_frame = QFrame()
        result_frame.setStyleSheet("QFrame { background: #e8f5e8; border-radius: 10px; padding: 15px; }")
        result_layout = QVBoxLayout(result_frame)
        pred_label = QLabel("Tespit Edilen Durum:")
        pred_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        result_layout.addWidget(pred_label)
        pred_value = QLabel(prediction)
        pred_value.setStyleSheet("font-size: 20px; font-weight: bold; color: #c0392b; margin: 5px;")
        result_layout.addWidget(pred_value)
        self.result_layout.addWidget(result_frame)
        prob_frame = QFrame()
        prob_frame.setStyleSheet("QFrame { background: #f8f9fa; border-radius: 10px; padding: 15px; }")
        prob_layout = QVBoxLayout(prob_frame)
        prob_title = QLabel("Tüm Sınıf Olasılıkları:")
        prob_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        prob_layout.addWidget(prob_title)
        for i, (label, prob) in enumerate(zip(self.label_names, probabilities)):
            color = "#c0392b" if label in prediction.split(', ') else "#7f8c8d"
            prob_item = QLabel(f"• {label}: %{prob:.1f}")
            prob_item.setStyleSheet(f"font-size: 12px; color: {color}; margin: 2px;")
            prob_layout.addWidget(prob_item)
        self.result_layout.addWidget(prob_frame)
        self.result_layout.addStretch()

    def show_error(self, error_message):
        # ... (Bu metot aynı, değişiklik yok)
        self.clear_layout(self.result_layout)
        error_label = QLabel(" Analiz Hatası")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c; margin: 20px;")
        self.result_layout.addWidget(error_label)
        error_detail = QLabel(error_message)
        error_detail.setAlignment(Qt.AlignCenter)
        error_detail.setWordWrap(True)
        error_detail.setStyleSheet("font-size: 12px; color: #c0392b; margin: 10px;")
        self.result_layout.addWidget(error_detail)