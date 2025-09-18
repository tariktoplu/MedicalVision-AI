# pages/multi_analysis_page_bt.py

import os
import json
from collections import Counter
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QTableWidgetItem, 
                             QHeaderView, QStyle, QLabel, QHBoxLayout, QFrame, QWidget)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

from .base_page import BaseMultiAnalysisPage
from workers import MultiAnalysisWorker

class MultiAnalysisPageBT(BaseMultiAnalysisPage):
    """
    BT için çoklu analiz sayfası.
    Dosya bazlı tahmin yapar ve sonuçları yarışma formatına uygun kaydeder.
    """
    def __init__(self, modality, models, device, label_names):
        super().__init__(modality, models, device, label_names)
        
        self.upload_file_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #3498db; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #5dade2; }")
        self.upload_folder_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #1abc9c; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #2fe2bf; }")
        
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Dosya Adı', 'Durum', 'Tahmin'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 3): header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    def get_worker_class(self):
        return MultiAnalysisWorker

    def populate_table(self):
        self.table.clearContents()
        self.table.setRowCount(len(self.file_paths))
        for i, file_path in enumerate(self.file_paths):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(file_path)))
            self.set_status_badge(i, "Bekliyor", "#f39c12")
            self.table.setItem(i, 2, QTableWidgetItem("-"))

    def start_analysis(self):
        if not self.file_paths: return
        self.set_ui_enabled(False)
        self.save_btn.setEnabled(False)
        self.prediction_results = {} 
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.right_stack.setCurrentWidget(self.initial_summary_widget)
        
        WorkerClass = self.get_worker_class()
        # Worker'a parent ataması yapmıyoruz, base class'taki safe_go_back halledecek
        self.analysis_worker = WorkerClass(self.models, self.device, self.file_paths, self.label_names, self.modality)
        
        self.analysis_worker.file_progress.connect(self.update_file_result)
        self.analysis_worker.file_error.connect(self.update_file_error)
        self.analysis_worker.all_finished.connect(self.on_analysis_finished)
        self.analysis_worker.start()

    def update_file_result(self, index, prediction, probabilities):
        self.set_status_badge(index, "Tamamlandı", "#27ae60")
        self.table.setItem(index, 2, QTableWidgetItem(prediction))
        
        pred_item = self.table.item(index, 2)
        pred_item.setForeground(QColor("#c0392b"))
        pred_item.setFont(QFont("Arial", -1, QFont.Bold))
        
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        
        self.prediction_results[self.file_paths[index]] = {"prediction": prediction}

    def on_analysis_finished(self):
        super().analysis_finished()

    def save_results_to_json(self, takim_adi, takim_id):
        if not self.prediction_results:
            QMessageBox.warning(self, "Kayıt Hatası", "Kaydedilecek sonuç bulunmuyor.")
            return
            
        modality_short = "CT"
        default_filename = f"{takim_id}_{takim_adi.replace(' ', '_')}_{modality_short}_Yarisma.json"
        save_path, _ = QFileDialog.getSaveFileName(self, "Sonuçları Kaydet", default_filename, "JSON Dosyaları (*.json)")
        
        if save_path:
            kunye = {"takim_adi": takim_adi, "takim_id": takim_id, "aciklama": f"{modality_short} Tahmin Verileri", "versiyon": "v1.0"}
            tahminler = []
            # Sıralı bir çıktı için dosya yollarını sırala
            for file_path in sorted(self.prediction_results.keys()):
                result = self.prediction_results[file_path]
                tahmin_obj = {"filename": os.path.basename(file_path), "stroke": 1 if result["prediction"] == "İnme" else 0, "stroke_type": 3}
                tahminler.append(tahmin_obj)
            
            final_json = {"kunye": kunye, "tahminler": tahminler}
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Başarılı", f"Sonuçlar '{os.path.basename(save_path)}' olarak kaydedildi.")
            except Exception as e:
                QMessageBox.critical(self, "Kayıt Hatası", f"Dosya kaydedilemedi:\n{str(e)}")

    def update_summary_panel(self):
        super().update_summary_panel()
        
        total_files = self.table.rowCount()
        statuses = [self.table.item(r, 1).text() for r in range(self.table.rowCount())]
        predictions = [self.table.item(r, 2).text() for r in range(self.table.rowCount()) if self.table.item(r, 1).text() == "Tamamlandı"]
        status_counts, prediction_counts = Counter(statuses), Counter(predictions)
        
        summary_title = QLabel("Analiz Sonuç Özeti")
        summary_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;")
        self.results_layout.addWidget(summary_title)
        
        self.results_layout.addWidget(self.create_summary_label("Toplam Dosya:", f"{total_files}"))
        self.results_layout.addWidget(self.create_summary_label("Başarılı:", f"{status_counts.get('Tamamlandı', 0)}", "#27ae60"))
        self.results_layout.addWidget(self.create_summary_label("Hatalı:", f"{status_counts.get('Hata', 0)}", "#e74c3c"))
        
        separator = QFrame(); separator.setFrameShape(QFrame.HLine); separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("margin-top: 10px; margin-bottom: 10px;")
        self.results_layout.addWidget(separator)
        
        prediction_title = QLabel("Tahmin Dağılımı")
        prediction_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #34495e; margin-bottom: 5px;")
        self.results_layout.addWidget(prediction_title)
        
        if not prediction_counts:
            no_preds_label = QLabel("Hiçbir başarılı tahmin bulunamadı."); no_preds_label.setStyleSheet("font-style: italic;")
            self.results_layout.addWidget(no_preds_label)
        else:
            for pred, count in prediction_counts.items():
                self.results_layout.addWidget(self.create_summary_label(f"{pred}:", f"{count} dosya"))
        self.results_layout.addStretch()