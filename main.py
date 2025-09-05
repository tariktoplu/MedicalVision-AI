# main.py

import sys
import os
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QStyle

from model import MR_ConvNeXt, BT_ConvNeXt
from pages import StartPage, AnalysisModePage, SingleAnalysisPage, MultiAnalysisPage

class MedicalImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medikal Görüntü Analiz Sistemi")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("QMainWindow { background-color: #ecf0f1; }")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mr_models = self._load_models_from_path("Models/MR", MR_ConvNeXt, "MR")
        self.bt_models = self._load_models_from_path("Models/BT", BT_ConvNeXt, "BT")
        
        self.label_names_mr = ['HiperakutAkut', 'Subakut', 'NormalKronik']
        self.label_names_bt = ['Sağlıklı', 'İnme']
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.start_page = StartPage()
        self.start_page.modality_selected.connect(self.show_mode_page)
        self.stack.addWidget(self.start_page)
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Hazır - {len(self.mr_models)} MR, {len(self.bt_models)} BT modeli yüklendi - Cihaz: {self.device}")
        
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
    
    def _load_models_from_path(self, model_path, model_class_or_function, label):
        models = []
        if not os.path.isdir(model_path):
            QMessageBox.critical(None, f"{label} Model Hatası", f"'{model_path}' klasörü bulunamadı!")
            return models
        try:
            model_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.pth'))]
            if not model_files:
                print(f"Uyarı: '{model_path}' klasöründe model dosyası bulunamadı.")
                return models

            for model_file in model_files:
                path = os.path.join(model_path, model_file)
                model = model_class_or_function()
                
                # --- ÇÖZÜM BURADA ---
                # Yeni PyTorch versiyonlarının güvenlik kontrolünü geçmek için
                # weights_only=False parametresini ekliyoruz.
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    state_dict_to_load = ckpt["state_dict"]
                else:
                    state_dict_to_load = ckpt
                
                # strict=False, esneklik sağlar.
                model.load_state_dict(state_dict_to_load, strict=False)

                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"'{path}' başarıyla yüklendi.")
            print(f"Toplam {len(models)} adet {label} modeli başarıyla yüklendi.")
        except Exception as e:
            QMessageBox.critical(None, f"{label} Model Yükleme Hatası", f"Modeller yüklenirken hata oluştu:\n{str(e)}")
        
        return models

    def show_analysis_page(self, modality, mode):
        if modality == "MR":
            models_to_use, labels_to_use = self.mr_models, self.label_names_mr
        elif modality == "BT":
            models_to_use, labels_to_use = self.bt_models, self.label_names_bt
        else: return

        if not models_to_use:
            QMessageBox.warning(self, "Model Eksik", f"'{modality}' için yüklenmiş model bulunamadı.")
            return

        if mode == "single":
            analysis_page = SingleAnalysisPage(modality, models_to_use, self.device, labels_to_use)
        else:
            analysis_page = MultiAnalysisPage(modality, models_to_use, self.device, labels_to_use)
        
        analysis_page.back_clicked.connect(self.show_start_page)
        self.stack.addWidget(analysis_page)
        self.stack.setCurrentWidget(analysis_page)
    
    def show_mode_page(self, modality):
        mode_page = AnalysisModePage(modality)
        mode_page.mode_selected.connect(self.show_analysis_page)
        mode_page.back_clicked.connect(self.show_start_page)
        self.stack.addWidget(mode_page)
        self.stack.setCurrentWidget(mode_page)
    
    def show_start_page(self):
        while self.stack.count() > 1:
            widget = self.stack.widget(1)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        self.stack.setCurrentWidget(self.start_page)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Medikal Görüntü Analiz Sistemi")
    app.setStyle('Fusion')
    window = MedicalImageAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()