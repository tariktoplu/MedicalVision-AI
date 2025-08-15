# pages/multi_analysis_page.py

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QScrollArea, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QProgressBar, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont 

# Projenin kök dizinindeki workers'ı import ediyoruz
from workers import MultiAnalysisWorker

class MultiAnalysisPage(QWidget):
    """Çoklu analiz sayfası"""
    # ... (Bu sınıfın tam ve güncel halini önceki yanıttan kopyalayın) ...
    # ... (QSplitter düzeltmesi dahil) ...
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
        back_btn.setStyleSheet("QPushButton { background: #95a5a6; color: white; border: none; border-radius: 8px; font-size: 12px; } QPushButton:hover { background: #7f8c8d; }")
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        title = QLabel(f"{self.modality} - Çoklu Analiz")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        left_layout = QVBoxLayout(left_panel)
        upload_section = QLabel(" Dosya Yükleme")
        upload_section.setAlignment(Qt.AlignCenter)
        upload_section.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(upload_section)
        self.upload_btn = QPushButton("Dosyaları Seç")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("QPushButton { background: #3498db; color: white; border: none; border-radius: 8px; font-size: 14px; } QPushButton:hover { background: #2980b9; }")
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
        self.table.setStyleSheet("QTableWidget { gridline-color: #bdc3c7; background-color: white; alternate-background-color: #f8f9fa; selection-background-color: #3498db; border: 1px solid #bdc3c7; border-radius: 5px; } QTableWidget::item { padding: 8px; border: none; } QHeaderView::section { background-color: #34495e; color: white; padding: 8px; border: none; font-weight: bold; }")
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 2px solid #bdc3c7; border-radius: 5px; text-align: center; font-weight: bold; } QProgressBar::chunk { background-color: #27ae60; border-radius: 3px; }")
        left_layout.addWidget(self.progress_bar)
        
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
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300]) 
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def upload_files(self):
        file_types = "Tüm Desteklenen (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Görüntü (*.png *.jpg *.jpeg *.bmp)"
        file_paths, _ = QFileDialog.getOpenFileNames(self, f"{self.modality} Dosyalarını Seç", "", file_types)
        if file_paths:
            self.file_paths = file_paths
            self.file_info.setText(f"{len(self.file_paths)} dosya seçildi.")
            self.populate_table()
            self.start_analysis()

    def populate_table(self):
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
        if not self.file_paths: return
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.worker = MultiAnalysisWorker(self.models, self.device, self.file_paths, self.label_names)
        self.worker.file_progress.connect(self.update_file_result)
        self.worker.file_error.connect(self.update_file_error)
        self.worker.all_finished.connect(self.analysis_finished)
        self.worker.start()

    def update_file_result(self, index, prediction, confidence, probabilities):
        self.table.setItem(index, 2, QTableWidgetItem("Tamamlandı"))
        self.table.setItem(index, 3, QTableWidgetItem(prediction))
        self.table.setItem(index, 4, QTableWidgetItem(f"{confidence:.1f}"))
        status_item = self.table.item(index, 2)
        status_item.setBackground(QColor("#27ae60"))
        status_item.setForeground(QColor("white"))
        pred_item = self.table.item(index, 3)
        pred_item.setForeground(QColor("#e74c3c"))
        pred_item.setFont(QFont("Arial", -1, QFont.Bold))
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def update_file_error(self, index, error_message):
        self.table.setItem(index, 2, QTableWidgetItem("Hata"))
        self.table.setItem(index, 3, QTableWidgetItem(error_message[:30] + "..."))
        self.table.setItem(index, 4, QTableWidgetItem("-"))
        status_item = self.table.item(index, 2)
        status_item.setBackground(QColor("#e74c3c"))
        status_item.setForeground(QColor("white"))
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def analysis_finished(self):
        self.progress_bar.setValue(len(self.file_paths))
        QMessageBox.information(self, "Analiz Tamamlandı", f"{len(self.file_paths)} dosyanın analizi tamamlandı!")

    def clear_files(self):
        self.file_paths.clear()
        self.table.setRowCount(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)