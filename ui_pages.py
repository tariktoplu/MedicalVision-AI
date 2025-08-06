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
    mode_selected = pyqtSignal(str, str)
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
            QPushButton { font-size: 12px; background: #95a5a6; color: white; border: none; border-radius: 8px; }
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
            QLabel { font-size: 24px; font-weight: bold; color: #2c3e50; margin: 20px; padding: 15px; background: #ecf0f1; border-radius: 10px; }
        """)
        layout.addWidget(title)
        
        # Mod butonları
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(50)
        
        # Tekli Analiz Butonu
        single_btn = QPushButton(" Tekli Analiz")
        single_btn.setFixedSize(200, 100)
        single_btn.setStyleSheet("""
            QPushButton { font-size: 16px; font-weight: bold; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #229954); color: white; border: none; border-radius: 15px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #58d68d, stop:1 #27ae60); }
        """)
        single_btn.clicked.connect(lambda: self.mode_selected.emit(self.modality, "single"))
        button_layout.addWidget(single_btn)
        
        # Çoklu Analiz Butonu
        multi_btn = QPushButton(" Çoklu Analiz")
        multi_btn.setFixedSize(200, 100)
        multi_btn.setStyleSheet("""
            QPushButton { font-size: 16px; font-weight: bold; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f39c12, stop:1 #e67e22); color: white; border: none; border-radius: 15px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8c471, stop:1 #f39c12); }
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
        self.setAcceptDrops(True)  # Sürükle-bırak özelliğini etkinleştir
        self.setup_ui()

    def clear_layout(self, layout):
        """Bir layout içindeki tüm widget'ları ve spacer'ları güvenle temizler."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Bir dosya widget üzerine sürüklendiğinde tetiklenir."""
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()
            self.preview_label.setStyleSheet("border: 3px dashed #3498db; border-radius: 8px;")

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Sürüklenen dosya widget'tan ayrıldığında tetiklenir."""
        if not self.current_file:
            self.preview_label.setStyleSheet("border: 2px dashed #bdc3c7; border-radius: 8px;")
        else:
            self.preview_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 8px;")

    def dropEvent(self, event: QDropEvent):
        """Dosya widget üzerine bırakıldığında tetiklenir."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            file_path = event.mimeData().urls()[0].toLocalFile()
            
            supported_formats = ('.dcm', '.png', '.jpg', '.jpeg', '.bmp')
            if file_path.lower().endswith(supported_formats):
                self.process_new_file(file_path)
            else:
                QMessageBox.warning(self, "Desteklenmeyen Dosya", "Lütfen desteklenen bir görüntü dosyası (.dcm, .png, .jpg) sürükleyin.")
                self.dragLeaveEvent(None) # Stili normale döndür

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        top_bar = QHBoxLayout()
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(80, 35)
        back_btn.setStyleSheet("""
            QPushButton { background: #95a5a6; color: white; border: none; border-radius: 8px; font-size: 12px; }
            QPushButton:hover { background: #7f8c8d; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        title = QLabel(f"{self.modality} - Tekli Analiz")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        content_layout = QHBoxLayout()
        
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        upload_section = QLabel(" Dosya Yükleme")
        upload_section.setAlignment(Qt.AlignCenter)
        upload_section.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(upload_section)
        
        self.upload_btn = QPushButton("Dosya Seç")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("""
            QPushButton { background: #3498db; color: white; border: none; border-radius: 8px; font-size: 14px; }
            QPushButton:hover { background: #2980b9; }
        """)
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
        
        content_layout.addWidget(left_panel)
        
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
        self.setLayout(main_layout)
    
    def process_new_file(self, file_path):
        """Yeni bir dosyayı işlemek için merkezi fonksiyon (sürükle-bırak ve buton için)."""
        self.current_file = file_path
        self.show_file_info(file_path)
        self.show_preview(file_path)
        self.analyze_file(file_path)
        
    def upload_file(self):
        """Dosya yükleme iletişim kutusunu açar."""
        file_types = "Tüm Desteklenen (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Görüntü (*.png *.jpg *.jpeg *.bmp)"
        file_path, _ = QFileDialog.getOpenFileName(self, f"{self.modality} Dosyası Seç", "", file_types)
        
        if file_path:
            self.process_new_file(file_path)
    
    def show_file_info(self, file_path):
        """Dosya bilgilerini gösterir."""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].upper()
        file_size = os.path.getsize(file_path) / 1024
        
        info_text = f"{file_name}\n{file_ext} • {file_size:.1f} KB"
        self.file_info.setText(info_text)
        self.file_info.setStyleSheet("margin: 10px; color: #2c3e50; font-weight: bold;")
    
    def show_preview(self, file_path):
        """Görüntü önizlemesini gösterir."""
        try:
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image_array = dcm.pixel_array
            else:
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            pil_image.thumbnail((280, 280), Image.Resampling.LANCZOS)
            
            qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 8px;")
            
        except Exception as e:
            self.preview_label.setText(f"Önizleme hatası:\n{str(e)}")
            self.preview_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 8px; color: red;")
            self.current_file = None # Hata durumunda dosyayı geçersiz kıl

    def analyze_file(self, file_path):
        """Dosya analizini başlatır."""
        # Sonuç alanını güvenli bir şekilde temizle
        self.clear_layout(self.result_layout)
        
        self.progress_label = QLabel(" Analiz başlatılıyor...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; color: #3498db; margin: 20px;")
        self.result_layout.addWidget(self.progress_label)
        
        self.worker = AnalysisWorker(self.models, self.device, file_path, self.label_names)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, message):
        """Analiz ilerlemesini günceller."""
        self.progress_label.setText(f" {message}")
    
    def show_results(self, prediction, confidence, probabilities):
        """Analiz sonuçlarını gösterir."""
        self.clear_layout(self.result_layout)

        success_label = QLabel(" Analiz Tamamlandı!")
        success_label.setAlignment(Qt.AlignCenter)
        success_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60; margin: 10px;")
        self.result_layout.addWidget(success_label)
        
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
        """Hata mesajını gösterir."""
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
        
        top_bar = QHBoxLayout()
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(80, 35)
        back_btn.setStyleSheet("""
            QPushButton { background: #95a5a6; color: white; border: none; border-radius: 8px; font-size: 12px; }
            QPushButton:hover { background: #7f8c8d; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        title = QLabel(f"{self.modality} - Çoklu Analiz")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        content_layout = QHBoxLayout()
        
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        upload_section = QLabel(" Dosya Yükleme")
        upload_section.setAlignment(Qt.AlignCenter)
        upload_section.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(upload_section)
        
        self.upload_btn = QPushButton("Dosyaları Seç")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("""
            QPushButton { background: #3498db; color: white; border: none; border-radius: 8px; font-size: 14px; }
            QPushButton:hover { background: #2980b9; }
        """)
        self.upload_btn.clicked.connect(self.upload_files)
        left_layout.addWidget(self.upload_btn)
        
        self.file_info = QLabel("Henüz dosya yüklenmedi")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setStyleSheet("margin: 10px; color: #7f8c8d;")
        left_layout.addWidget(self.file_info)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Dosya Adı', 'Format', 'Durum', 'Tahmin', 'Güven (%)'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 5): header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        self.table.setStyleSheet("""
            QTableWidget { gridline-color: #bdc3c7; background-color: white; alternate-background-color: #f8f9fa; selection-background-color: #3498db; border: 1px solid #bdc3c7; border-radius: 5px; }
            QTableWidget::item { padding: 8px; border: none; }
            QHeaderView::section { background-color: #34495e; color: white; padding: 8px; border: none; font-weight: bold; }
        """)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #bdc3c7; border-radius: 5px; text-align: center; font-weight: bold; }
            QProgressBar::chunk { background-color: #27ae60; border-radius: 3px; }
        """)
        left_layout.addWidget(self.progress_bar)
        content_layout.addWidget(left_panel)
        
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        right_layout = QVBoxLayout(right_panel)
        
        result_title = QLabel(" Analiz Özeti")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        right_layout.addWidget(result_title)
        
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_area.setStyleSheet("border: none;")
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)
        self.result_area.setWidget(self.result_widget)
        
        initial_msg = QLabel("Analiz için dosyaları yükleyin")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setStyleSheet("color: #95a5a6; font-size: 14px; margin: 50px;")
        self.result_layout.addWidget(initial_msg)
        
        right_layout.addWidget(self.result_area)
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
    
    def upload_files(self):
        """Çoklu dosya yükleme."""
        file_types = "Tüm Desteklenen (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Görüntü (*.png *.jpg *.jpeg *.bmp)"
        file_paths, _ = QFileDialog.getOpenFileNames(self, f"{self.modality} Dosyalarını Seç", "", file_types)
        
        if file_paths:
            self.file_paths = file_paths
            self.file_info.setText(f"{len(self.file_paths)} dosya seçildi.")
            self.populate_table()
            self.start_analysis()
    
    def populate_table(self):
        """Tabloyu dosyalarla doldurur."""
        self.table.setRowCount(len(self.file_paths))
        for i, file_path in enumerate(self.file_paths):
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].upper()
            
            self.table.setItem(i, 0, QTableWidgetItem(file_name))
            self.table.setItem(i, 1, QTableWidgetItem(file_ext))
            self.table.setItem(i, 2, QTableWidgetItem("Bekliyor"))
            self.table.setItem(i, 3, QTableWidgetItem("-"))
            self.table.setItem(i, 4, QTableWidgetItem("-"))
            
            status_item = self.table.item(i, 2)
            status_item.setBackground(QColor("#f39c12"))
            status_item.setForeground(QColor("white"))
    
    def start_analysis(self):
        """Analizi başlatır."""
        if not self.file_paths:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        
        self.worker = MultiAnalysisWorker(self.models, self.device, self.file_paths, self.label_names)
        self.worker.file_progress.connect(self.update_file_result)
        self.worker.file_error.connect(self.update_file_error)
        self.worker.all_finished.connect(self.analysis_finished)
        self.worker.start()
    
    def update_file_result(self, index, prediction, confidence, probabilities):
        """Tablodaki bir dosyanın sonucunu günceller."""
        self.table.setItem(index, 2, QTableWidgetItem("Tamamlandı"))
        self.table.setItem(index, 3, QTableWidgetItem(prediction))
        self.table.setItem(index, 4, QTableWidgetItem(f"{confidence:.1f}"))
        
        status_item = self.table.item(index, 2)
        status_item.setBackground(QColor("#27ae60"))
        pred_item = self.table.item(index, 3)
        pred_item.setForeground(QColor("#e74c3c"))
        pred_item.setFont(QFont("Arial", -1, QFont.Bold))
        
        self.progress_bar.setValue(self.progress_bar.value() + 1)
    
    def update_file_error(self, index, error_message):
        """Tablodaki bir dosyanın hata durumunu günceller."""
        self.table.setItem(index, 2, QTableWidgetItem("Hata"))
        self.table.setItem(index, 3, QTableWidgetItem(error_message[:30] + "..."))
        self.table.setItem(index, 4, QTableWidgetItem("-"))
        
        status_item = self.table.item(index, 2)
        status_item.setBackground(QColor("#e74c3c"))
        
        self.progress_bar.setValue(self.progress_bar.value() + 1)
    
    def analysis_finished(self):
        """Analiz tamamlandığında çağrılır."""
        self.progress_bar.setValue(len(self.file_paths))
        QMessageBox.information(self, "Analiz Tamamlandı", f"{len(self.file_paths)} dosyanın analizi tamamlandı!")