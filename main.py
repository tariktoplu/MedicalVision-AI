# main.py

import sys
import os
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QStyle

# Doğru model sınıflarını ve sayfaları import et
from model import MR_ConvNeXt, BT_ConvNeXt
from pages import StartPage, AnalysisModePage, SingleAnalysisPage, MultiAnalysisPageMR, MultiAnalysisPageBT

class MedicalImageAnalyzer(QMainWindow):
    """Ana uygulama sınıfı"""
    
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
        
        self.page_history = []
        
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
                
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    state_dict_to_load = ckpt["state_dict"]
                else:
                    state_dict_to_load = ckpt
                
                model.load_state_dict(state_dict_to_load, strict=False)

                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"'{path}' başarıyla yüklendi.")
            print(f"Toplam {len(models)} adet {label} modeli başarıyla yüklendi.")
        except Exception as e:
            QMessageBox.critical(None, f"{label} Model Yükleme Hatası", f"Modeller yüklenirken hata oluştu:\n{str(e)}")
        
        return models

    def show_mode_page(self, modality):
        current_page = self.stack.currentWidget()
        if current_page not in self.page_history:
            self.page_history.append(current_page)

        mode_page = AnalysisModePage(modality)
        mode_page.mode_selected.connect(self.show_analysis_page)
        mode_page.back_clicked.connect(self.go_back)
        self.stack.addWidget(mode_page)
        self.stack.setCurrentWidget(mode_page)
    
    def show_analysis_page(self, modality, mode):
        current_page = self.stack.currentWidget()
        if current_page not in self.page_history:
            self.page_history.append(current_page)

        models_to_use, labels_to_use = (self.mr_models, self.label_names_mr) if modality == "MR" else (self.bt_models, self.label_names_bt)

        if not models_to_use:
            QMessageBox.warning(self, "Model Eksik", f"'{modality}' için yüklenmiş model bulunamadı.")
            self.go_back()
            return

        analysis_page = None
        if mode == "single":
            analysis_page = SingleAnalysisPage(modality, models_to_use, self.device, labels_to_use)
        else: # multi modu
            if modality == "MR":
                analysis_page = MultiAnalysisPageMR(modality, models_to_use, self.device, labels_to_use)
            elif modality == "BT":
                analysis_page = MultiAnalysisPageBT(modality, models_to_use, self.device, labels_to_use)
        
        if analysis_page:
            analysis_page.back_clicked.connect(self.go_back)
            self.stack.addWidget(analysis_page)
            self.stack.setCurrentWidget(analysis_page)
    
    def go_back(self):
        if self.page_history:
            page_to_remove = self.stack.currentWidget()
            # İşçilerin sinyallerini güvenle kes
            if hasattr(page_to_remove, 'disconnect_worker_signals'):
                page_to_remove.disconnect_worker_signals()
            
            self.stack.removeWidget(page_to_remove)
            page_to_remove.deleteLater()
            
            prev_page = self.page_history.pop()
            self.stack.setCurrentWidget(prev_page)
        else:
            self.show_start_page()

    def show_start_page(self):
        while self.stack.count() > 1:
            widget = self.stack.widget(1)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        
        self.page_history.clear()
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