# pages/multi_analysis_page.py

import os
from collections import Counter
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QScrollArea, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QProgressBar, QSplitter, QStyle, QStackedWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QFont, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QIcon

from workers import MultiAnalysisWorker

class MultiAnalysisPage(QWidget):
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality
        self.models = models
        self.device = device
        self.label_names = label_names
        self.file_paths = []
        self.setAcceptDrops(True)
        self.setup_ui()
    
    # ... (setup_ui ve diğer metotlar) ...
    def start_analysis(self):
        if not self.file_paths: return
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.right_stack.setCurrentWidget(self.initial_summary_widget)
        
        # --- DÜZELTME BURADA ---
        self.worker = MultiAnalysisWorker(self.models, self.device, self.file_paths, self.label_names, self.modality)
        
        self.worker.file_progress.connect(self.update_file_result)
        self.worker.file_error.connect(self.update_file_error)
        self.worker.all_finished.connect(self.analysis_finished)
        self.worker.start()

    # ... (Geri kalan tüm metotlar aynı kalabilir) ...
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.left_panel.setStyleSheet(self.style_sheet_drop_active)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.left_panel.setStyleSheet(self.style_sheet_default)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            dropped_files = [url.toLocalFile() for url in event.mimeData().urls()]
            supported_formats = ('.dcm', '.png', '.jpg', '.jpeg', '.bmp')
            valid_files = [f for f in dropped_files if f.lower().endswith(supported_formats)]
            
            if valid_files:
                self.process_new_files(valid_files)
            else:
                QMessageBox.warning(self, "Desteklenmeyen Dosyalar", "Desteklenen bir görüntü formatı seçin.")
        
        self.left_panel.setStyleSheet(self.style_sheet_default)

    def setup_ui(self):
        self.style_sheet_default = "QFrame { background-color: white; border-radius: 10px; }"
        self.style_sheet_drop_active = "QFrame { background-color: #eaf5ff; border: 2px dashed #3498db; border-radius: 10px; }"
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        top_bar = QHBoxLayout()
        back_icon = self.style().standardIcon(QStyle.SP_ArrowLeft)
        back_btn = QPushButton(back_icon, " Geri")
        back_btn.setFixedSize(100, 40)
        back_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #7f8c8d; color: white; border: none; border-radius: 8px; } QPushButton:hover { background-color: #95a5a6; }")
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)

        title = QLabel(f"{self.modality} - Çoklu Analiz")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50; margin-left: 15px;")
        top_bar.addWidget(title)
        
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        
        self.left_panel = QFrame()
        self.left_panel.setStyleSheet(self.style_sheet_default)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        upload_area_label = QLabel("Dosyaları Buraya Sürükleyin")
        upload_area_label.setAlignment(Qt.AlignCenter)
        upload_area_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #34495e; margin-bottom: 10px;")
        left_layout.addWidget(upload_area_label)
        
        button_layout = QHBoxLayout()
        upload_icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        self.upload_btn = QPushButton(upload_icon, " Dosya Seç...")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #3498db; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #5dade2; }")
        self.upload_btn.clicked.connect(self.upload_files_from_dialog)
        button_layout.addWidget(self.upload_btn)

        clear_icon = self.style().standardIcon(QStyle.SP_TrashIcon)
        self.clear_btn = QPushButton(clear_icon, " Listeyi Temizle")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #e74c3c; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #ec7063; }")
        self.clear_btn.clicked.connect(self.clear_files)
        button_layout.addWidget(self.clear_btn)
        left_layout.addLayout(button_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Dosya Adı', 'Durum', 'Tahmin', 'Güven (%)'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4): header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.setStyleSheet("""
            QTableWidget { gridline-color: #e0e0e0; border: 1px solid #e0e0e0; border-radius: 8px; }
            QHeaderView::section { background-color: #f2f2f2; padding: 8px; border: none; font-weight: bold; color: #34495e; }
        """)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; border-radius: 8px; background-color: #e0e0e0; text-align: center; font-weight: bold; color: #34495e; }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #2ecc71); border-radius: 8px; }
        """)
        left_layout.addWidget(self.progress_bar)

        right_panel = QFrame()
        right_panel.setStyleSheet(self.style_sheet_default)
        right_layout = QVBoxLayout(right_panel)
        
        self.right_stack = QStackedWidget()
        
        self.initial_summary_widget = QWidget()
        initial_layout = QVBoxLayout(self.initial_summary_widget)
        initial_layout.setAlignment(Qt.AlignCenter)
        summary_icon = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        summary_icon.setPixmap(icon.pixmap(QSize(80, 80)))
        summary_icon.setAlignment(Qt.AlignCenter)
        summary_icon.setStyleSheet("margin-bottom: 20px;")
        initial_layout.addWidget(summary_icon)
        initial_msg = QLabel("Analiz Sonuçları Özeti Burada Görüntülenecek")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setWordWrap(True)
        initial_msg.setStyleSheet("color: #7f8c8d; font-size: 18px;")
        initial_layout.addWidget(initial_msg)
        
        self.results_summary_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_summary_widget)
        self.results_layout.setContentsMargins(20, 20, 20, 20)
        self.results_layout.setAlignment(Qt.AlignTop)

        self.right_stack.addWidget(self.initial_summary_widget)
        self.right_stack.addWidget(self.results_summary_widget)
        right_layout.addWidget(self.right_stack)

        splitter.addWidget(self.left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 400])
        splitter.setStyleSheet("QSplitter::handle { background-color: #ecf0f1; } QSplitter::handle:hover { background-color: #bdc3c7; }")
        main_layout.addWidget(splitter, 1)

    def process_new_files(self, file_paths):
        self.file_paths = file_paths
        self.populate_table()
        self.start_analysis()

    def upload_files_from_dialog(self):
        file_types = "Tüm Desteklenen (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Görüntü (*.png *.jpg *.jpeg *.bmp)"
        file_paths, _ = QFileDialog.getOpenFileNames(self, f"{self.modality} Dosyalarını Seç", "", file_types)
        if file_paths:
            self.process_new_files(file_paths)

    def populate_table(self):
        self.table.clearContents()
        self.table.setRowCount(len(self.file_paths))
        for i, file_path in enumerate(self.file_paths):
            file_name = os.path.basename(file_path)
            self.table.setItem(i, 0, QTableWidgetItem(file_name))
            self.set_status_badge(i, "Bekliyor", "#f39c12")
            self.table.setItem(i, 2, QTableWidgetItem("-"))
            self.table.setItem(i, 3, QTableWidgetItem("-"))

    def set_status_badge(self, row, text, color):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(QColor(color))
        item.setForeground(QColor("white"))
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        self.table.setItem(row, 1, item)

    def update_file_result(self, index, prediction, confidence, probabilities):
        self.set_status_badge(index, "Tamamlandı", "#27ae60")
        self.table.setItem(index, 2, QTableWidgetItem(prediction))
        self.table.setItem(index, 3, QTableWidgetItem(f"{confidence:.1f}"))
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def update_file_error(self, index, error_message):
        self.set_status_badge(index, "Hata", "#e74c3c")
        self.table.setItem(index, 2, QTableWidgetItem("Hata oluştu"))
        self.table.setItem(index, 3, QTableWidgetItem("-"))
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def analysis_finished(self):
        self.progress_bar.setFormat("Analiz tamamlandı!")
        self.update_summary_panel()
        self.right_stack.setCurrentWidget(self.results_summary_widget)

    def update_summary_panel(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        total_files = self.table.rowCount()
        statuses = [self.table.item(row, 1).text() for row in range(total_files)]
        predictions = [self.table.item(row, 2).text() for row in range(total_files) if self.table.item(row, 1).text() == "Tamamlandı"]

        status_counts = Counter(statuses)
        prediction_counts = Counter(predictions)
        
        summary_title = QLabel("Analiz Sonuç Özeti")
        summary_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;")
        self.results_layout.addWidget(summary_title)
        
        self.results_layout.addWidget(self.create_summary_label("Toplam Dosya:", f"{total_files}"))
        self.results_layout.addWidget(self.create_summary_label("Başarılı:", f"{status_counts.get('Tamamlandı', 0)}", "#27ae60"))
        self.results_layout.addWidget(self.create_summary_label("Hatalı:", f"{status_counts.get('Hata', 0)}", "#e74c3c"))
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("margin-top: 10px; margin-bottom: 10px;")
        self.results_layout.addWidget(separator)
        
        prediction_title = QLabel("Tahmin Dağılımı")
        prediction_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #34495e; margin-bottom: 5px;")
        self.results_layout.addWidget(prediction_title)
        
        if not prediction_counts:
            no_preds_label = QLabel("Hiçbir başarılı tahmin bulunamadı.")
            no_preds_label.setStyleSheet("font-style: italic;")
            self.results_layout.addWidget(no_preds_label)
        else:
            for pred, count in prediction_counts.items():
                self.results_layout.addWidget(self.create_summary_label(f"{pred}:", f"{count} dosya"))
        
        self.results_layout.addStretch()

    def create_summary_label(self, key_text, value_text, value_color=None):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        key_label = QLabel(key_text)
        key_label.setStyleSheet("font-weight: bold;")
        
        value_label = QLabel(value_text)
        if value_color:
            value_label.setStyleSheet(f"font-weight: bold; color: {value_color};")
        
        layout.addWidget(key_label)
        layout.addWidget(value_label)
        layout.addStretch()
        
        return widget

    def clear_files(self):
        self.file_paths.clear()
        self.table.clearContents()
        self.table.setRowCount(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.right_stack.setCurrentWidget(self.initial_summary_widget)