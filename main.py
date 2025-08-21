# main.py

import sys
import os
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QStyle

from model import MedNet, BT_ConvNeXt
from pages import StartPage, AnalysisModePage, SingleAnalysisPage, MultiAnalysisPage

class MedicalImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medikal Görüntü Analiz Sistemi")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("QMainWindow { background-color: #ecf0f1; }")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mr_models = self._load_models_from_path(
            model_path="Models/MR", 
            model_class=MedNet,
            label="MR"
        )
        self.bt_models = self._load_models_from_path(
            model_path="Models/BT",
            model_class=BT_ConvNeXt,
            label="BT"
        )
        
        self.label_names_mr = ['HiperakutAkut', 'Subakut', 'NormalKronik']
        self.label_names_bt = ['Sağlıklı', 'İnme', 'Diğer'] 
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.start_page = StartPage()
        self.start_page.modality_selected.connect(self.show_mode_page)
        self.stack.addWidget(self.start_page)
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Hazır - {len(self.mr_models)} MR, {len(self.bt_models)} BT modeli yüklendi - Cihaz: {self.device}")
        
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
    
    def _load_models_from_path(self, model_path, model_class, label):
        models = []
        if not os.path.isdir(model_path):
            QMessageBox.critical(None, f"{label} Model Hatası", f"'{model_path}' klasörü bulunamadı!")
            return models

        try:
            model_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.pth'))]
            if not model_files:
                print(f"Uyarı: '{model_path}' klasöründe yüklenecek model dosyası bulunamadı.")
                return models

            for model_file in model_files:
                path = os.path.join(model_path, model_file)
                
                # 1. Modeli oluştur
                model = model_class()
                
                # 2. Ağırlıkları yükle
                model.load_state_dict(torch.load(path, map_location=self.device))
                
                # --- ÇOK ÖNEMLİ ADIM ---
                # 3. Eğer bu bir BT modeli ise, uygulamaya adapte et
                if label == "BT":
                    model.adapt_to_application()

                # 4. Modeli cihaza taşı ve eval moduna al
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"'{path}' başarıyla yüklendi ve uygulamaya uyarlandı.")
            
            print(f"Toplam {len(models)} adet {label} modeli başarıyla yüklendi.")
        except Exception as e:
            QMessageBox.critical(None, f"{label} Model Yükleme Hatası", f"Modeller yüklenirken hata oluştu:\n{str(e)}")
        
        return models

    def show_analysis_page(self, modality, mode):
        if modality == "MR":
            models_to_use = self.mr_models
            labels_to_use = self.label_names_mr
        elif modality == "BT":
            models_to_use = self.bt_models
            labels_to_use = self.label_names_bt
        else:
            return

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
    
    # ... (Geri kalan metotlar aynı) ...
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