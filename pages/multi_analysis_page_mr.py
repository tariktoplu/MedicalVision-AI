# pages/multi_analysis_page_mr.py

import os
import json
from collections import defaultdict, Counter
import torch
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QProgressBar, QSplitter, QStyle, 
                             QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import (QColor, QFont, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QIcon,
                         QCursor)

# Temel sınıfı ve ortak yardımcı sınıfları import et
from .base_page import BaseMultiAnalysisPage, KunyeDialog, FolderScannerWorker
from workers import AnalysisWorker

# MR'a özel hasta bazlı worker
class MRAggregationWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, models, device, file_paths, label_names):
        super().__init__()
        self.models = models
        self.device = device
        self.file_paths = file_paths
        self.label_names = label_names

    def get_patient_id_from_path(self, path, common_path):
        """Dosya yolundan hasta kimliğini çıkarmak için sağlam bir yöntem."""
        try:
            # Ortak yola göre göreceli yolu bul
            relative_path = os.path.relpath(os.path.dirname(path), common_path)
            # Eğer dosya doğrudan ortak yolun içindeyse, ortak yolun adını PID olarak kullan
            if relative_path == '.':
                return os.path.basename(common_path)
            # Aksi takdirde, göreceli yolun ilk klasörünü PID olarak al
            return relative_path.split(os.sep)[0]
        except (IndexError, ValueError):
            # Yukarıdakiler başarısız olursa, dosyanın kendi klasörünün adını kullan
            return os.path.basename(os.path.dirname(path))

    def run(self):
        try:
            patient_slice_probs = defaultdict(list)
            
            if not self.file_paths:
                self.finished.emit({})
                return
            
            # Tüm dosyaların en ortak üst klasörünü bul
            common_path = os.path.dirname(self.file_paths[0])
            if len(self.file_paths) > 1:
                common_path = os.path.commonpath(self.file_paths)

            for i, file_path in enumerate(self.file_paths):
                # AnalysisWorker'daki preprocess'i kullanarak tensörü oluştur
                temp_worker = AnalysisWorker(self.models, self.device, file_path, self.label_names, 'MR')
                tensor = temp_worker.preprocess_image(file_path).to(self.device)
                
                with torch.no_grad():
                    # Birden fazla model varsa (ensemble), ortalamasını al
                    logits_list = [model(tensor) for model in self.models]
                    avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                    probabilities = torch.sigmoid(avg_logits).cpu().numpy()[0]
                    
                    pid = self.get_patient_id_from_path(file_path, common_path)
                    patient_slice_probs[pid].append(probabilities)
                    
                self.progress.emit(i + 1, len(self.file_paths))

            # Hasta bazlı nihai tahminleri oluştur
            patient_final_preds = {}
            for pid, all_probs in patient_slice_probs.items():
                # Hastanın tüm dilimleri için olasılıkların ortalamasını al
                avg_patient_probs = np.mean(all_probs, axis=0)
                
                # Multi-label mantığını uygula
                ML_THRESH = 0.5
                preds_binary = (avg_patient_probs >= ML_THRESH).astype(int)
                
                # Eğer hiçbir sınıf eşiği geçemezse, en yüksek olasılığa sahip olanı seç
                if np.sum(preds_binary) == 0:
                    top_idx = np.argmax(avg_patient_probs)
                    preds_binary = np.zeros_like(preds_binary)
                    preds_binary[top_idx] = 1
                    
                patient_final_preds[pid] = preds_binary.tolist()

            self.finished.emit(patient_final_preds)
        except Exception as e:
            self.error.emit(f"Hasta bazlı analiz hatası: {str(e)}")


class MultiAnalysisPageMR(BaseMultiAnalysisPage):
    """
    MR için çoklu analiz sayfası.
    Hasta bazlı tahmin yapar ve sonuçları yarışma formatına uygun kaydeder.
    """
    def __init__(self, modality, models, device, label_names):
        super().__init__(modality, models, device, label_names)
        
        # Arayüzü MR'a özel hale getir
        self.title_label.setText(f"{self.modality} - Çoklu Analiz (Hasta Bazlı)")
        self.upload_area_label.setText("Hasta Klasörlerini Buraya Sürükleyin")
        self.upload_file_btn.hide()
        self.upload_folder_btn.setText(" Hasta Klasörlerini Seç...")
        
        # Tabloyu MR'a özel hale getir
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Hasta ID', 'Tahmin Edilen Durum(lar)'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
    
    def get_worker_class(self):
        return MRAggregationWorker

    def start_analysis(self):
        if not self.file_paths: return
        self.set_ui_enabled(False)
        self.save_btn.setEnabled(False)
        self.prediction_results = {}
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Dilimler analiz ediliyor: %v/%m")
        
        WorkerClass = self.get_worker_class()
        self.analysis_worker = WorkerClass(self.models, self.device, self.file_paths, self.label_names)
        self.analysis_worker.progress.connect(lambda current, total: self.progress_bar.setValue(current))
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.start()

    def on_analysis_error(self, error_message):
        self.set_ui_enabled(True)
        self.progress_bar.setFormat("Hata oluştu!")
        QMessageBox.critical(self, "Analiz Hatası", error_message)
        
    def on_analysis_finished(self, patient_preds):
        self.set_ui_enabled(True)
        if patient_preds:
            self.save_btn.setEnabled(True)
        self.progress_bar.setFormat("Analiz tamamlandı!")
        self.prediction_results = patient_preds
        
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

        self.update_summary_panel()
        self.right_stack.setCurrentWidget(self.results_summary_widget)

    def save_results_to_json(self, takim_adi, takim_id):
        if not self.prediction_results:
            QMessageBox.warning(self, "Kayıt Hatası", "Kaydedilecek sonuç yok.")
            return
            
        default_filename = f"{takim_id}_{takim_adi.replace(' ', '_')}_MR_Yarisma.json"
        save_path, _ = QFileDialog.getSaveFileName(self, "Sonuçları Kaydet", default_filename, "JSON Dosyaları (*.json)")

        if save_path:
            kunye = {"takim_adi": takim_adi, "takim_id": takim_id, "aciklama": "MR Tahmin Verileri", "versiyon": "v2.0"}
            tahminler = []
            for pid, pred_vector in sorted(self.prediction_results.items()):
                tahmin_obj = {
                    "PatientID": pid,
                    "hyperacute_acute": pred_vector[0],
                    "subacute": pred_vector[1],
                    "normal_chronic": pred_vector[2]
                }
                tahminler.append(tahmin_obj)
            
            final_json = {"kunye": kunye, "tahminler": tahminler}
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Başarılı", f"Sonuçlar '{os.path.basename(save_path)}' olarak kaydedildi.")
            except Exception as e:
                QMessageBox.critical(self, "Kayıt Hatası", f"Dosya kaydedilemedi:\n{str(e)}")

    def update_summary_panel(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        
        total_patients = len(self.prediction_results)
        prediction_counts = Counter()
        for pid, pred_vector in self.prediction_results.items():
            labels = [self.label_names[j] for j, val in enumerate(pred_vector) if val == 1]
            if not labels:
                prediction_counts["Tanımlanamadı"] += 1
            else:
                for label in labels:
                    prediction_counts[label] += 1

        summary_title = QLabel("Hasta Bazlı Sonuç Özeti")
        summary_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;")
        self.results_layout.addWidget(summary_title)
        self.results_layout.addWidget(self.create_summary_label("Toplam Hasta:", f"{total_patients}"))
        
        separator = QFrame(); separator.setFrameShape(QFrame.HLine); separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("margin-top: 10px; margin-bottom: 10px;")
        self.results_layout.addWidget(separator)
        
        prediction_title = QLabel("Tahmin Dağılımı (Hasta Sayısı)")
        prediction_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #34495e; margin-bottom: 5px;")
        self.results_layout.addWidget(prediction_title)
        
        if not prediction_counts:
            no_preds_label = QLabel("Hiçbir başarılı tahmin bulunamadı."); no_preds_label.setStyleSheet("font-style: italic;")
            self.results_layout.addWidget(no_preds_label)
        else:
            for pred, count in prediction_counts.items():
                self.results_layout.addWidget(self.create_summary_label(f"{pred}:", f"{count} hasta"))
        
        self.results_layout.addStretch()

    def clear_files(self):
        super().clear_files()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Hasta ID', 'Tahmin Edilen Durum(lar)'])

    def populate_table(self):
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Taranan Dosyalar', 'Durum'])
        self.table.setRowCount(len(self.file_paths))
        for i, path in enumerate(self.file_paths):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(path)))
            self.table.setItem(i, 1, QTableWidgetItem("Analiz için hazır"))

    # Bu metotlar base sınıfta tanımlı ve bu alt sınıfta kullanılmıyor.
    def update_file_result(self, index, prediction, probabilities):
        pass