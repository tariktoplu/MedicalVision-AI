# pages/start_page.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal

class StartPage(QWidget):
    """Başlangıç sayfası"""
    modality_selected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Başlık
        title = QLabel(" Medikal Görüntü Analiz Sistemi")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 28px; font-weight: bold; color: #2c3e50; margin: 20px; padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white; border-radius: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Modalite seçim başlığı
        subtitle = QLabel("Görüntü Modalitesi Seçiniz:")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; font-weight: bold; color: #34495e; margin: 10px;")
        layout.addWidget(subtitle)
        
        # Buton container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(50)
        
        # BT Butonu
        bt_btn = QPushButton("BT")
        bt_btn.setFixedSize(200, 100)
        bt_btn.setStyleSheet("""
            QPushButton { font-size: 20px; font-weight: bold; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9); color: white; border: none; border-radius: 15px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1b4f72); }
        """)
        bt_btn.clicked.connect(lambda: self.modality_selected.emit("BT"))
        button_layout.addWidget(bt_btn)
        
        # MR Butonu
        mr_btn = QPushButton("MR")
        mr_btn.setFixedSize(200, 100)
        mr_btn.setStyleSheet("""
            QPushButton { font-size: 20px; font-weight: bold; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b); color: white; border: none; border-radius: 15px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #e74c3c); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c0392b, stop:1 #922b21); }
        """)
        mr_btn.clicked.connect(lambda: self.modality_selected.emit("MR"))
        button_layout.addWidget(mr_btn)
        
        layout.addWidget(button_container)
        layout.addStretch()
        
        self.setLayout(layout)