# pages/multi_analysis_page_mr.py

import os
import json
from collections import defaultdict, Counter
# --- YENİ EKLENEN IMPORTLAR ---
import torch
import numpy as np
# -----------------------------
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QScrollArea, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QProgressBar, QSplitter, QStyle, 
                             QStackedWidget, QApplication, QLineEdit, QFormLayout, QDialog,
                             QDialogButtonBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import (QColor, QFont, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QIcon,
                         QCursor)

from workers import AnalysisWorker

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

class MRAggregationWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, models, device, file_paths, label_names):
        super().__init__()
        self.models = models
        self.device = device
        self.file_paths = file_paths
        self.label_names = label_names

    def run(self):
        try:
            patient_slice_preds = defaultdict(list)
            
            # --- DÜZELTME: Hasta ID'sini bulmak için daha sağlam bir yöntem ---
            # En başta, tüm dosya yollarının ortak en uzun yolunu bulalım. Bu bizim ana veri klasörümüzdür.
            if not self.file_paths:
                self.finished.emit({})
                return
            
            common_path = os.path.commonpath(self.file_paths)

            for i, file_path in enumerate(self.file_paths):
                temp_worker = AnalysisWorker(self.models, self.device, file_path, self.label_names, 'MR')
                tensor = temp_worker.preprocess_image(file_path).to(self.device)
                
                with torch.no_grad():
                    avg_logits = torch.mean(torch.stack([model(tensor) for model in self.models]), dim=0)
                    probabilities = torch.sigmoid(avg_logits).cpu().numpy()[0]
                    
                    # PatientID'yi dosya yolundan çıkar
                    try:
                        # Ortak yoldan sonraki ilk klasör adını PID olarak al
                        relative_path = os.path.relpath(file_path, common_path)
                        pid = relative_path.split(os.sep)[0]
                    except (IndexError, ValueError):
                        pid = f"unknown_patient_{i}" # Eğer yol beklenmedikse
                        
                    patient_slice_preds[pid].append(probabilities)
                self.progress.emit(i + 1)

            patient_final_preds = {}
            for pid, all_probs in patient_slice_preds.items():
                final_pred_vector = [0, 0, 0]
                avg_patient_probs = np.mean(all_probs, axis=0)
                
                ML_THRESH = 0.5
                preds_binary = (avg_patient_probs >= ML_THRESH).astype(int)
                
                if np.sum(preds_binary) == 0:
                    top_prediction_idx = np.argmax(avg_patient_probs)
                    preds_binary = np.zeros_like(preds_binary)
                    preds_binary[top_prediction_idx] = 1
                    
                patient_final_preds[pid] = preds_binary.tolist()

            self.finished.emit(patient_final_preds)
        except Exception as e:
            self.error.emit(f"Hasta bazlı analiz hatası: {str(e)}")


class MultiAnalysisPageMR(QWidget):
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality; self.models = models; self.device = device; self.label_names = label_names
        self.file_paths = []; self.scanner_worker = None; self.patient_predictions = {}
        self.setAcceptDrops(True); self.setup_ui()
    
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
        self.style_sheet_default = "QFrame { background-color: white; border-radius: 10px; }"
        self.style_sheet_drop_active = "QFrame { background-color: #eaf5ff; border: 2px dashed #3498db; border-radius: 10px; }"
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        top_bar = QHBoxLayout()
        back_btn = QPushButton(self.style().standardIcon(QStyle.SP_ArrowLeft), " Geri")
        back_btn.setFixedSize(100, 40)
        back_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #7f8c8d; color: white; border: none; border-radius: 8px; } QPushButton:hover { background-color: #95a5a6; }")
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        title = QLabel(f"{self.modality} - Çoklu Analiz (Hasta Bazlı)")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50; margin-left: 15px;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        splitter = QSplitter(Qt.Horizontal)
        self.left_panel = QFrame()
        self.left_panel.setStyleSheet(self.style_sheet_default)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        upload_area_label = QLabel("Hasta Klasörlerini Buraya Sürükleyin")
        upload_area_label.setAlignment(Qt.AlignCenter)
        upload_area_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #34495e; margin-bottom: 10px;")
        left_layout.addWidget(upload_area_label)
        top_button_layout = QHBoxLayout()
        upload_folder_icon = self.style().standardIcon(QStyle.SP_DirOpenIcon)
        self.upload_folder_btn = QPushButton(upload_folder_icon, " Hasta Klasörlerini Seç...")
        self.upload_folder_btn.setFixedHeight(40)
        self.upload_folder_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #1abc9c; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #2fe2bf; }")
        self.upload_folder_btn.clicked.connect(self.upload_folder_from_dialog)
        top_button_layout.addWidget(self.upload_folder_btn)
        left_layout.addLayout(top_button_layout)
        bottom_button_layout = QHBoxLayout()
        self.clear_btn = QPushButton(self.style().standardIcon(QStyle.SP_TrashIcon), " Listeyi Temizle")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #e74c3c; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #ec7063; }")
        self.clear_btn.clicked.connect(self.clear_files)
        bottom_button_layout.addWidget(self.clear_btn)
        self.save_btn = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), " Sonuçları Kaydet")
        self.save_btn.setFixedHeight(40)
        self.save_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #27ae60; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #2ecc71; }")
        self.save_btn.clicked.connect(self.open_kunye_dialog_and_save)
        self.save_btn.setEnabled(False)
        bottom_button_layout.addWidget(self.save_btn)
        left_layout.addLayout(bottom_button_layout)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Hasta ID', 'Tahmin Edilen Durum(lar)'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        right_panel = QFrame()
        right_panel.setStyleSheet(self.style_sheet_default)
        right_layout = QVBoxLayout(right_panel)
        initial_msg = QLabel("Hasta bazlı analiz sonuçları tamamlandığında burada görünecektir.")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setWordWrap(True)
        right_layout.addWidget(initial_msg)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 400])
        main_layout.addWidget(splitter, 1)

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
        self.upload_folder_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)

    def process_new_files(self, file_paths):
        self.file_paths = file_paths
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(['Taranan Dosyalar', 'Durum'])
        self.table.setRowCount(len(file_paths))
        for i, path in enumerate(file_paths):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(path)))
            self.table.setItem(i, 1, QTableWidgetItem("Analiz için hazır"))
        self.start_analysis()

    def upload_folder_from_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, f"Hasta Klasörlerinin Bulunduğu Ana Klasörü Seç")
        if folder_path:
            self.handle_paths([folder_path])

    def start_analysis(self):
        if not self.file_paths: return
        self.set_ui_enabled(False)
        self.save_btn.setEnabled(False)
        self.patient_predictions = {}
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Dilimler analiz ediliyor: %p%")
        
        self.worker = MRAggregationWorker(self.models, self.device, self.file_paths, self.label_names)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_error(self, error_message):
        self.set_ui_enabled(True)
        QMessageBox.critical(self, "Analiz Hatası", error_message)
        
    def on_analysis_finished(self, patient_preds):
        self.set_ui_enabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setFormat("Analiz tamamlandı!")
        self.patient_predictions = patient_preds
        
        self.table.clearContents()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Hasta ID', 'Tahmin Edilen Durum(lar)'])
        self.table.setRowCount(len(patient_preds))
        
        for i, (pid, pred_vector) in enumerate(sorted(patient_preds.items())):
            labels = [self.label_names[j] for j, val in enumerate(pred_vector) if val == 1]
            if not labels: labels = ["Tanımlanamadı"]
            
            self.table.setItem(i, 0, QTableWidgetItem(pid))
            self.table.setItem(i, 1, QTableWidgetItem(", ".join(labels)))
        self.table.resizeColumnsToContents()

    def open_kunye_dialog_and_save(self):
        dialog = KunyeDialog("TUSEB_SYZ_MR", "0123456", self)
        if dialog.exec_() == QDialog.Accepted:
            takim_adi, takim_id = dialog.get_data()
            if not takim_adi or not takim_id:
                 QMessageBox.warning(self, "Eksik Bilgi", "Takım Adı ve ID boş bırakılamaz.")
                 return
            self.save_results_to_json(takim_adi, takim_id)

    def save_results_to_json(self, takim_adi, takim_id):
        if not self.patient_predictions:
            QMessageBox.warning(self, "Kayıt Hatası", "Kaydedilecek sonuç yok.")
            return
            
        default_filename = f"{takim_id}_{takim_adi.replace(' ', '_')}_MR_Yarisma.json"
        save_path, _ = QFileDialog.getSaveFileName(self, "Sonuçları Kaydet", default_filename, "JSON Dosyaları (*.json)")

        if save_path:
            kunye = {"kunye": {"takim_adi": takim_adi, "takim_id": takim_id, "aciklama": "MR Tahmin Verileri", "versiyon": "v2.0"}}
            tahminler = []
            for pid, pred_vector in sorted(self.patient_predictions.items()):
                tahmin_obj = {
                    "PatientID": pid,
                    "hyperacute_acute": pred_vector[0],
                    "subacute": pred_vector[1],
                    "normal_chronic": pred_vector[2]
                }
                tahminler.append(tahmin_obj)
            
            final_data = kunye
            final_data["tahminler"] = tahminler
            
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Başarılı", f"Sonuçlar '{os.path.basename(save_path)}' olarak kaydedildi.")
            except Exception as e:
                QMessageBox.critical(self, "Kayıt Hatası", f"Dosya kaydedilemedi:\n{str(e)}")

    def clear_files(self):
        self.file_paths.clear()
        self.patient_predictions.clear()
        self.save_btn.setEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(['Hasta ID', 'Tahmin Edilen Durum(lar)'])
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)