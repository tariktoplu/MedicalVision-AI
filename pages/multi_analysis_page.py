# pages/multi_analysis_page.py

import os
import json
from collections import Counter
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QScrollArea, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QProgressBar, QSplitter, QStyle, 
                             QStackedWidget, QApplication, QLineEdit, QFormLayout, QDialog,
                             QDialogButtonBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import (QColor, QFont, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QIcon,
                         QCursor)

from workers import MultiAnalysisWorker

# --- Künye Bilgilerini Almak İçin Diyalog Penceresi ---
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
            
            unique_sorted_files = sorted(list(set(all_files)))
            self.finished.emit(unique_sorted_files)
        except Exception as e:
            self.error.emit(f"Klasör taranırken bir hata oluştu: {str(e)}")


class MultiAnalysisPage(QWidget):
    back_clicked = pyqtSignal()
    
    def __init__(self, modality, models, device, label_names):
        super().__init__()
        self.modality = modality
        self.models = models
        self.device = device
        self.label_names = label_names
        self.file_paths = []
        self.scanner_worker = None
        self.prediction_results = []
        self.json_template = None
        self.json_template_path = ""
        self.setAcceptDrops(True)
        self.setup_ui()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.left_panel.setStyleSheet(self.style_sheet_drop_active)
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
        upload_area_label = QLabel("Dosyaları veya Klasörleri Buraya Sürükleyin\nveya Butonları Kullanın")
        upload_area_label.setAlignment(Qt.AlignCenter)
        upload_area_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #34495e; margin-bottom: 10px;")
        left_layout.addWidget(upload_area_label)
        
        top_button_layout = QHBoxLayout()
        self.upload_template_btn = QPushButton(self.style().standardIcon(QStyle.SP_FileIcon), " JSON Şablonu Yükle...")
        self.upload_template_btn.setFixedHeight(40)
        self.upload_template_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #9b59b6; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #af7ac5; }")
        self.upload_template_btn.clicked.connect(self.load_json_template)
        top_button_layout.addWidget(self.upload_template_btn)

        upload_folder_icon = self.style().standardIcon(QStyle.SP_DirOpenIcon)
        self.upload_folder_btn = QPushButton(upload_folder_icon, " Klasör Seç...")
        self.upload_folder_btn.setFixedHeight(40)
        self.upload_folder_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #1abc9c; color: white; border-radius: 8px; padding: 5px; } QPushButton:hover { background-color: #2fe2bf; }")
        self.upload_folder_btn.clicked.connect(self.upload_folder_from_dialog)
        top_button_layout.addWidget(self.upload_folder_btn)
        left_layout.addLayout(top_button_layout)
        
        bottom_button_layout = QHBoxLayout()
        clear_icon = self.style().standardIcon(QStyle.SP_TrashIcon)
        self.clear_btn = QPushButton(clear_icon, " Listeyi Temizle")
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
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Dosya Adı', 'Durum', 'Tahmin'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 3): header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
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
        summary_icon = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        summary_icon.setPixmap(icon.pixmap(QSize(80, 80)))
        summary_icon.setAlignment(Qt.AlignCenter)
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
            self.json_template = None # Klasör taraması yapıldığında şablonu sıfırla
            self.process_new_files(found_files)
        else:
            QMessageBox.warning(self, "Dosya Bulunamadı", "Seçilen konumlarda desteklenen formatta (.dcm, .png, .jpg, .jpeg) bir görüntü bulunamadı.")
    
    def on_scanning_error(self, error_message):
        QApplication.restoreOverrideCursor()
        self.set_ui_enabled(True)
        QMessageBox.critical(self, "Tarama Hatası", error_message)

    def set_ui_enabled(self, enabled):
        self.upload_template_btn.setEnabled(enabled)
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

    def load_json_template(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Yarışma JSON Şablonunu Seç", "", "JSON Dosyaları (*.json)")
        if not json_path: return
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_template = json.load(f)
            self.json_template_path = json_path
        except Exception as e:
            QMessageBox.critical(self, "JSON Okuma Hatası", f"JSON şablonu okunurken bir hata oluştu:\n{str(e)}")
            return
        data_folder = QFileDialog.getExistingDirectory(self, f"Yarışma Veri Seti Klasörünü Seç ({self.modality})")
        if not data_folder:
            self.json_template = None
            return
        self.populate_from_template(data_folder)

    def populate_from_template(self, data_folder):
        if not self.json_template: return
        template_filenames = {item['filename'] for item in self.json_template.get('tahminler', [])}
        found_files_map = {os.path.basename(f): f for f in self.find_all_files(data_folder)}
        found_filenames = set(found_files_map.keys())
        missing_in_json = found_filenames - template_filenames
        if missing_in_json:
            reply = QMessageBox.question(self, "Eksik Dosyalar Bulundu",
                                       f"{len(missing_in_json)} dosya klasörde bulundu ancak JSON şablonunda eksik.\nBu dosyalar listeye eklenip analiz edilsin mi?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                for fname in sorted(list(missing_in_json)):
                    new_entry = self.create_zero_prediction(fname)
                    self.json_template['tahminler'].append(new_entry)
        final_file_paths = [path for fname, path in found_files_map.items() if fname in {item['filename'] for item in self.json_template.get('tahminler', [])}]
        self.process_new_files(final_file_paths)

    def find_all_files(self, data_folder):
        all_files = []
        supported_extensions = ('.dcm', '.png', '.jpg', '.jpeg')
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    all_files.append(os.path.join(root, file))
        return all_files

    def create_zero_prediction(self, filename):
        if self.modality == 'MR':
            return {"filename": filename, "hyperacute_acute": 0, "subacute": 0, "normal_chronic": 0}
        else:
            return {"filename": filename, "stroke": 0, "stroke_type": 3}
            
    def populate_table(self):
        self.table.clearContents()
        self.table.setRowCount(len(self.file_paths))
        for i, file_path in enumerate(self.file_paths):
            file_name = os.path.basename(file_path)
            self.table.setItem(i, 0, QTableWidgetItem(file_name))
            self.set_status_badge(i, "Bekliyor", "#f39c12")
            self.table.setItem(i, 2, QTableWidgetItem("-"))

    def set_status_badge(self, row, text, color):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(QColor(color))
        item.setForeground(QColor("white"))
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        self.table.setItem(row, 1, item)
    
    def start_analysis(self):
        if not self.file_paths: return
        self.set_ui_enabled(False)
        self.save_btn.setEnabled(False)
        self.prediction_results = []
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.right_stack.setCurrentWidget(self.initial_summary_widget)
        self.worker = MultiAnalysisWorker(self.models, self.device, self.file_paths, self.label_names, self.modality)
        self.worker.file_progress.connect(self.update_file_result)
        self.worker.file_error.connect(self.update_file_error)
        self.worker.all_finished.connect(self.analysis_finished)
        self.worker.start()

    def update_file_result(self, index, prediction, probabilities):
        self.set_status_badge(index, "Tamamlandı", "#27ae60")
        self.table.setItem(index, 2, QTableWidgetItem(prediction))
        pred_item = self.table.item(index, 2)
        pred_item.setForeground(QColor("#c0392b"))
        pred_item.setFont(QFont("Arial", -1, QFont.Bold))
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        self.prediction_results.append({"file_path": self.file_paths[index], "prediction": prediction})

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

    def open_kunye_dialog_and_save(self):
        takim_adi, takim_id = "TUSEB_SYZ_" + self.modality, "000000"
        if self.json_template and 'kunye' in self.json_template:
            takim_adi = self.json_template['kunye'].get('takim_adi', takim_adi)
            takim_id = self.json_template['kunye'].get('takim_id', takim_id)
        
        dialog = KunyeDialog(takim_adi, takim_id, self)
        if dialog.exec_() == QDialog.Accepted:
            final_takim_adi, final_takim_id = dialog.get_data()
            if not final_takim_adi or not final_takim_id:
                 QMessageBox.warning(self, "Eksik Bilgi", "Takım Adı ve ID boş bırakılamaz.")
                 return
            self.save_results_to_json(final_takim_adi, final_takim_id)

    def save_results_to_json(self, takim_adi, takim_id):
        if not self.prediction_results and not self.json_template:
            QMessageBox.warning(self, "Kayıt Hatası", "Kaydedilecek sonuç bulunmuyor.")
            return
            
        final_json = {}
        if self.json_template:
            final_json = self.json_template.copy()
            results_map = {os.path.basename(res["file_path"]): res["prediction"] for res in self.prediction_results}
            for item in final_json.get('tahminler', []):
                if item['filename'] in results_map:
                    prediction = results_map[item['filename']]
                    if self.modality == 'MR':
                        item["hyperacute_acute"] = 1 if "HiperakutAkut" in prediction else 0
                        item["subacute"] = 1 if "Subakut" in prediction else 0
                        item["normal_chronic"] = 1 if "NormalKronik" in prediction else 0
                    elif self.modality == 'BT':
                        item["stroke"] = 1 if "İnme" in prediction else 0
                        item["stroke_type"] = 3
        else: # Şablon yok, sıfırdan oluştur
            tahminler = []
            if self.modality == 'MR':
                for res in self.prediction_results:
                    tahminler.append(self.create_zero_prediction(os.path.basename(res['file_path'])))
            elif self.modality == 'BT':
                for res in self.prediction_results:
                    tahminler.append(self.create_zero_prediction(os.path.basename(res['file_path'])))
            final_json['tahminler'] = tahminler
            
        final_json['kunye'] = {
            "takim_adi": takim_adi, "takim_id": takim_id,
            "aciklama": f"{self.modality} Tahmin Verileri", "versiyon": "v3.0"
        }
            
        default_filename = f"{takim_id}_{takim_adi.replace(' ', '_')}_{self.modality}_Yarisma.json"
        save_path, _ = QFileDialog.getSaveFileName(self, "Sonuçları Kaydet", default_filename, "JSON Dosyaları (*.json)")
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Başarılı", f"Sonuçlar başarıyla '{os.path.basename(save_path)}' dosyasına kaydedildi.")
            except Exception as e:
                QMessageBox.critical(self, "Kayıt Hatası", f"Dosya kaydedilirken bir hata oluştu:\n{str(e)}")

    def update_summary_panel(self):
        # ... (Bu metot aynı, değişiklik yok)
        pass
        
    def create_summary_label(self, key_text, value_text, value_color=None):
        # ... (Bu metot aynı, değişiklik yok)
        pass
        
    def clear_files(self):
        self.file_paths.clear()
        self.prediction_results.clear()
        self.save_btn.setEnabled(False)
        self.json_template = None
        self.json_template_path = ""
        self.table.clearContents()
        self.table.setRowCount(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.right_stack.setCurrentWidget(self.initial_summary_widget)