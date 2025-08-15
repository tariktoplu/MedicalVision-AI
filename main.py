# medical_analyzer_project/main.py

import sys
import os
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QStyle
from PyQt5.QtGui import QPalette, QColor

# Diğer Python dosyalarımızdan gerekli sınıfları içe aktaralım
from model import MR3DCNN_LSTM_Attention
from pages import StartPage, AnalysisModePage, SingleAnalysisPage, MultiAnalysisPage

class MedicalImageAnalyzer(QMainWindow):
    """Ana uygulama sınıfı"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medikal Görüntü Analiz Sistemi")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Modern stil
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
        """)
        
        # Model yükleme
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models()
        self.label_names = ['HiperakutAkut', 'Subakut', 'NormalKronik']
        
        # Stack widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Sayfaları oluştur
        self.start_page = StartPage()
        self.start_page.modality_selected.connect(self.show_mode_page)
        self.stack.addWidget(self.start_page)
        
        # Durum çubuğu
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Hazır - {len(self.models)} model yüklendi - Cihaz: {self.device}")
        
        # İkon ayarla (opsiyonel)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
    
    def load_models(self):
        """5 fold modelini yükle"""
        models = []
        model_dir = "Models"
        
        try:
            for fold in range(5):
                model_path = f"{model_dir}/best_model_fold_{fold}.pt"
                if os.path.exists(model_path):
                    model = MR3DCNN_LSTM_Attention()
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    models.append(model)
                    print(f"DEBUG: Model fold {fold} yüklendi. Veri Tipi: {next(model.parameters()).dtype}, Cihaz: {next(model.parameters()).device}")
                    
                else:
                    print(f"Uyarı: {model_path} bulunamadı")
            
            if not models:
                # QMessageBox'in düzgün çalışması için ana uygulama döngüsü başlamadan önce
                # geçici bir QApplication oluşturmak gerekebilir, ama burada QMainWindow içindeyiz.
                QMessageBox.critical(None, "Model Hatası", 
                                   "Hiçbir model dosyası bulunamadı!\n"
                                   "Models/ klasöründe model dosyalarını kontrol edin.")
            else:
                print(f"Toplam {len(models)} model başarıyla yüklendi")
                
        except Exception as e:
            QMessageBox.critical(None, "Model Yükleme Hatası", 
                               f"Modeller yüklenirken hata oluştu:\n{str(e)}")
        
        return models
    
    def show_mode_page(self, modality):
        """Mod seçim sayfasını göster"""
        mode_page = AnalysisModePage(modality)
        mode_page.mode_selected.connect(self.show_analysis_page)
        mode_page.back_clicked.connect(self.show_start_page)
        
        self.stack.addWidget(mode_page)
        self.stack.setCurrentWidget(mode_page)
    
    def show_analysis_page(self, modality, mode):
        """Analiz sayfasını göster"""
        if mode == "single":
            analysis_page = SingleAnalysisPage(modality, self.models, self.device, self.label_names)
        else:
            analysis_page = MultiAnalysisPage(modality, self.models, self.device, self.label_names)
        
        analysis_page.back_clicked.connect(self.show_start_page)
        
        self.stack.addWidget(analysis_page)
        self.stack.setCurrentWidget(analysis_page)
    
    def show_start_page(self):
        """Başlangıç sayfasını göster"""
        # Eski sayfaları temizle (memory leak önleme)
        while self.stack.count() > 1:
            widget = self.stack.widget(1)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        
        self.stack.setCurrentWidget(self.start_page)

def main():
    """Ana fonksiyon"""
    app = QApplication(sys.argv)
    
    # Uygulama bilgileri
    app.setApplicationName("Medikal Görüntü Analiz Sistemi")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Medical AI Solutions")
    
    # Modern stil
    app.setStyle('Fusion')
    
    # Dark palette (opsiyonel)
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # app.setPalette(palette)
    
    # Ana pencereyi oluştur ve göster
    window = MedicalImageAnalyzer()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()