# medical_analyzer_project/ui_pages.py

import os
import pydicom
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Diğer dosyalardan gerekli sınıfları içe aktar
from workers import AnalysisWorker, MultiAnalysisWorker

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