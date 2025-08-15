# pages/analysis_mode_page.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal

class AnalysisModePage(QWidget):
    """Analiz modu seçim sayfası"""
    mode_selected = pyqtSignal(str, str)
    back_clicked = pyqtSignal()
    
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Geri butonu
        back_btn = QPushButton(" Geri")
        back_btn.setFixedSize(100, 40)
        back_btn.setStyleSheet("""
            QPushButton { font-size: 12px; background: #95a5a6; color: white; border: none; border-radius: 8px; }
            QPushButton:hover { background: #7f8c8d; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        
        back_layout = QHBoxLayout()
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        
        # Başlık
        title = QLabel(f"{self.modality} Analiz Modu")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel { font-size: 24px; font-weight: bold; color: #2c3e50; margin: 20px; padding: 15px; background: #ecf0f1; border-radius: 10px; }
        """)
        layout.addWidget(title)
        
        # Mod butonları
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(50)
        
        # Tekli Analiz Butonu
        single_btn = QPushButton(" Tekli Analiz")
        single_btn.setFixedSize(200, 100)
        single_btn.setStyleSheet("""
            QPushButton { font-size: 16px; font-weight: bold; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #229954); color: white; border: none; border-radius: 15px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #58d68d, stop:1 #27ae60); }
        """)
        single_btn.clicked.connect(lambda: self.mode_selected.emit(self.modality, "single"))
        button_layout.addWidget(single_btn)
        
        # Çoklu Analiz Butonu
        multi_btn = QPushButton(" Çoklu Analiz")
        multi_btn.setFixedSize(200, 100)
        multi_btn.setStyleSheet("""
            QPushButton { font-size: 16px; font-weight: bold; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f39c12, stop:1 #e67e22); color: white; border: none; border-radius: 15px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8c471, stop:1 #f39c12); }
        """)
        multi_btn.clicked.connect(lambda: self.mode_selected.emit(self.modality, "multi"))
        button_layout.addWidget(multi_btn)
        
        layout.addWidget(button_container)
        layout.addStretch()
        
        self.setLayout(layout)