# pages/base_page.py

import os
from collections import Counter
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QScrollArea, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QProgressBar, QSplitter, QStyle, 
                             QStackedWidget, QApplication, QLineEdit, QFormLayout, QDialog,
                             QDialogButtonBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import (QColor, QFont, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QIcon,
                         QCursor)

# --- Yardımcı Sınıflar (Tüm sayfalarda ortak) ---
class KunyeDialog(QDialog):
    def __init__(self, takim_adi, takim_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Künye Bilgilerini Girin")
        self.takim_adi_input = QLineEdit(takim_adi)
        self.takim_id_input = QLineEdit(takim_id)
        form_layout = QFormLayout(self)
        form_layout.addRow("Takım Adı:", self.takim_adi_input)
        form_layout.addRow("Takım ID:", self.takim_id_input)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        form_layout.addRow(button_box)
    def get_data(self):
        return self.takim_adi_input.text(), self.takim_id_input.text()

class FolderScannerWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, paths_to_scan):
        super().__init__()
        self.paths = paths_to_scan
        self.supported_extensions = ('.dcm', '.png', '.jpg', '.jpeg')
    def run(self):
        try:
            all_files = []
            for path in self.paths:
                if os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file in files:
                            if file.lower().endswith(self.supported_extensions):
                                all_files.append(os.path.join(root, file))
                elif os.path.isfile(path) and path.lower().endswith(self.supported_extensions):
                    all_files.append(path)
            self.finished.emit(sorted(list(set(all_files))))
        except Exception as e:
            self.error.emit(f"Klasör taranırken hata: {str(e)}")


class BaseMultiAnalysisPage(QWidget):
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality
        self.models = models
        self.device = device
        self.label_names = label_names
        self.file_paths = []
        self.scanner_worker = None
        self.analysis_worker = None
        self.prediction_results = {}
        self.setAcceptDrops(True)
        self.setup_ui()
    
    def disconnect_worker_signals(self):
        worker = None
        if hasattr(self, 'scanner_worker') and self.scanner_worker and self.scanner_worker.isRunning():
            worker = self.scanner_worker
        elif hasattr(self, 'analysis_worker') and self.analysis_worker and self.analysis_worker.isRunning():
            worker = self.analysis_worker
        
        if worker:
            try:
                # Olası tüm sinyalleri disconnect et
                if hasattr(worker, 'progress'): worker.progress.disconnect()
                if hasattr(worker, 'file_progress'): worker.file_progress.disconnect()
                if hasattr(worker, 'finished'): worker.finished.disconnect()
                if hasattr(worker, 'error'): worker.error.disconnect()
                if hasattr(worker, 'all_finished'): worker.all_finished.disconnect()
            except TypeError:
                pass

    def safe_go_back(self):
        self.disconnect_worker_signals()
        QApplication.restoreOverrideCursor()
        self.back_clicked.emit()
            
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction(); self.left_panel.setStyleSheet(self.style_sheet_drop_active)
            
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.left_panel.setStyleSheet(self.style_sheet_default)
        
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.handle_paths(paths)
        self.left_panel.setStyleSheet(self.style_sheet_default)

    def setup_ui(self):
        from ui.theme import Colors
        self.style_sheet_default = f"QFrame {{ background-color: {Colors.BG_CARD}; border-radius: 8px; border: 1px solid {Colors.BORDER}; }}"
        self.style_sheet_drop_active = f"QFrame {{ background-color: {Colors.BG_CARD2}; border: 1px dashed {Colors.ACCENT}; border-radius: 8px; }}"
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ── Üst bar ──────────────────────────────────────────────────────
        top_bar_frame = QFrame()
        top_bar_frame.setFixedHeight(52)
        top_bar_frame.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_PRIMARY};"
            f" border-bottom:1px solid {Colors.BORDER}; border-radius:0; border:none;"
            f" border-bottom:1px solid {Colors.BORDER}; }}"
        )
        top_bar = QHBoxLayout(top_bar_frame)
        top_bar.setContentsMargins(24, 0, 24, 0)
        
        back_btn = QPushButton("← Geri")
        back_btn.setObjectName("btn_back")
        back_btn.setFixedHeight(32)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.safe_go_back)
        top_bar.addWidget(back_btn)
        top_bar.addSpacing(16)
        
        self.title_label = QLabel(f"{self.modality}  •  Çoklu Analiz")
        self.title_label.setStyleSheet(
            f"font-size:16px; font-weight:600; color:{Colors.TEXT_PRIMARY};"
        )
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        main_layout.addWidget(top_bar_frame)
        
        # ── İçerik alanı ─────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet(f"background:{Colors.BG_PRIMARY};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 12)
        content_layout.setSpacing(10)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(f"QSplitter {{ background:{Colors.BG_PRIMARY}; }}")
        self.left_panel = QFrame()
        self.left_panel.setStyleSheet(self.style_sheet_default)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        
        self.upload_area_label = QLabel("Dosyaları veya Klasörleri Sürükleyin")
        self.upload_area_label.setAlignment(Qt.AlignCenter)
        self.upload_area_label.setStyleSheet(
            f"font-size:14px; font-weight:600; color:{Colors.TEXT_SECONDARY}; margin-bottom:10px;"
        )
        left_layout.addWidget(self.upload_area_label)
        
        top_button_layout = QHBoxLayout()
        self.upload_file_btn = QPushButton("Dosya Seç")
        self.upload_file_btn.setObjectName("btn_primary")
        self.upload_file_btn.setFixedHeight(36)
        self.upload_file_btn.setCursor(Qt.PointingHandCursor)
        self.upload_file_btn.clicked.connect(self.upload_files_from_dialog)
        top_button_layout.addWidget(self.upload_file_btn)
        
        self.upload_folder_btn = QPushButton("Klasör Seç")
        self.upload_folder_btn.setFixedHeight(36)
        self.upload_folder_btn.setCursor(Qt.PointingHandCursor)
        self.upload_folder_btn.clicked.connect(self.upload_folder_from_dialog)
        top_button_layout.addWidget(self.upload_folder_btn)
        left_layout.addLayout(top_button_layout)
        
        bottom_button_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Temizle")
        self.clear_btn.setFixedHeight(36)
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_files)
        bottom_button_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Sonuçları Kaydet")
        self.save_btn.setObjectName("btn_success")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self.open_kunye_dialog_and_save)
        self.save_btn.setEnabled(False)
        bottom_button_layout.addWidget(self.save_btn)
        left_layout.addLayout(bottom_button_layout)
        
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        right_panel = QFrame()
        right_panel.setStyleSheet(self.style_sheet_default)
        right_layout = QVBoxLayout(right_panel)
        
        self.right_stack = QStackedWidget()
        self.initial_summary_widget = QWidget()
        initial_layout = QVBoxLayout(self.initial_summary_widget)
        initial_layout.setAlignment(Qt.AlignCenter)
        initial_msg = QLabel("Analiz sonuçları burada görüntülenecek")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setWordWrap(True)
        initial_msg.setStyleSheet(
            f"color:{Colors.TEXT_MUTED}; font-size:12px;"
        )
        initial_layout.addWidget(initial_msg)
        
        self.results_summary_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_summary_widget)
        self.results_layout.setContentsMargins(16, 16, 16, 16)
        self.results_layout.setAlignment(Qt.AlignTop)
        
        self.right_stack.addWidget(self.initial_summary_widget)
        self.right_stack.addWidget(self.results_summary_widget)
        right_layout.addWidget(self.right_stack)
        
        splitter.addWidget(self.left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 400])
        content_layout.addWidget(splitter, 1)
        main_layout.addWidget(content, 1)

    def handle_paths(self, paths):
        self.set_ui_enabled(False)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.scanner_worker = FolderScannerWorker(paths)
        self.scanner_worker.finished.connect(self.on_scanning_finished)
        self.scanner_worker.error.connect(self.on_scanning_error)
        self.scanner_worker.start()

    def on_scanning_finished(self, found_files):
        QApplication.restoreOverrideCursor()
        self.set_ui_enabled(True)
        if found_files:
            self.process_new_files(found_files)
        else:
            QMessageBox.warning(self, "Dosya Bulunamadı", "Desteklenen formatta dosya bulunamadı.")
    
    def on_scanning_error(self, error_message):
        QApplication.restoreOverrideCursor()
        self.set_ui_enabled(True)
        QMessageBox.critical(self, "Tarama Hatası", error_message)

    def set_ui_enabled(self, enabled):
        self.upload_file_btn.setEnabled(enabled)
        self.upload_folder_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)

    def process_new_files(self, file_paths):
        self.file_paths = file_paths
        self.populate_table()
        self.start_analysis()

    def upload_files_from_dialog(self):
        file_types = "Tüm Desteklenen Görüntüler (*.dcm *.png *.jpg *.jpeg);;Tüm Dosyalar (*)"
        file_paths, _ = QFileDialog.getOpenFileNames(self, f"{self.modality} Dosyalarını Seç", "", file_types)
        if file_paths:
            self.handle_paths(file_paths)

    def upload_folder_from_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, f"{self.modality} Klasörünü Seç")
        if folder_path:
            self.handle_paths([folder_path])

    def set_status_badge(self, row, text, color):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(QColor(color))
        item.setForeground(QColor("white"))
        font = QFont(); font.setBold(True); item.setFont(font)
        self.table.setItem(row, 1, item)
    
    def on_analysis_error(self, error_message):
        self.set_ui_enabled(True)
        self.progress_bar.setFormat("Hata oluştu!")
        QMessageBox.critical(self, "Analiz Hatası", error_message)

    def open_kunye_dialog_and_save(self):
        dialog = KunyeDialog("TUSEB_SYZ_" + self.modality, "987654", self)
        if dialog.exec_() == QDialog.Accepted:
            takim_adi, takim_id = dialog.get_data()
            if not takim_adi or not takim_id:
                 QMessageBox.warning(self, "Eksik Bilgi", "Takım Adı ve ID boş bırakılamaz.")
                 return
            self.save_results_to_json(takim_adi, takim_id)

    def update_summary_panel(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
            
    def create_summary_label(self, key_text, value_text, value_color=None):
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0,0,0,0)
        key_label = QLabel(key_text); key_label.setStyleSheet("font-weight: bold;")
        value_label = QLabel(value_text)
        if value_color: value_label.setStyleSheet(f"font-weight: bold; color: {value_color};")
        layout.addWidget(key_label); layout.addWidget(value_label); layout.addStretch()
        return widget
        
    def clear_files(self):
        self.disconnect_worker_signals()
        self.file_paths.clear()
        self.prediction_results.clear()
        self.save_btn.setEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.right_stack.setCurrentWidget(self.initial_summary_widget)

    # --- YENİ EKLENEN ORTAK METOTLAR ---
    def update_file_error(self, index, error_message):
        self.set_status_badge(index, "Hata", "#e74c3c")
        self.table.setItem(index, 2, QTableWidgetItem("Hata oluştu"))
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def analysis_finished(self):
        self.set_ui_enabled(True)
        if self.prediction_results:
            self.save_btn.setEnabled(True)
        self.progress_bar.setFormat("Analiz tamamlandı!")
        self.update_summary_panel()
        self.right_stack.setCurrentWidget(self.results_summary_widget)

    # --- ALT SINIFLARIN EZMESİ GEREKEN SOYUT METOTLAR ---
    def get_worker_class(self): raise NotImplementedError
    def start_analysis(self): raise NotImplementedError
    def update_file_result(self, index, prediction, probabilities): raise NotImplementedError
    def on_analysis_finished(self): self.analysis_finished()
    def save_results_to_json(self, takim_adi, takim_id): raise NotImplementedError
    def populate_table(self): raise NotImplementedError