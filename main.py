# main.py
# Minimalist Dark Medical Dashboard — Sidebar'sız, temiz navigasyon
# Turan YZ — NeuroViva AI

import sys
import os
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap

from model import MR_ConvNeXt, BT_ConvNeXt
from pages import (
    StartPage, AnalysisModePage, SingleAnalysisPage,
    MultiAnalysisPageMR, MultiAnalysisPageBT
)
from ui.theme import DARK_THEME, Colors, LOGO_DARK


# ─────────────────────────────────────────────────────────────────────────────
# Ana Pencere — Minimalist, sidebar'sız
# ─────────────────────────────────────────────────────────────────────────────
class MedicalImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroViva AI — Turan YZ")
        self.setGeometry(100, 60, 1100, 760)
        self.setMinimumSize(900, 600)

        # Pencere ikonu
        if os.path.exists(LOGO_DARK):
            self.setWindowIcon(QIcon(LOGO_DARK))

        # Tema uygula
        self.setStyleSheet(DARK_THEME)

        # Cihaz + Modeller
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mr_models = self._load_models("Models/MR", MR_ConvNeXt, "MR")
        self.bt_models = self._load_models("Models/BT", BT_ConvNeXt, "BT")

        self.label_names_mr = ['HiperakutAkut', 'Subakut', 'NormalKronik']
        self.label_names_bt = ['Sağlıklı', 'İnme']

        self.page_history: list = []

        self._build_ui()

        # Durum çubuğu
        mr_count = len(self.mr_models)
        bt_count = len(self.bt_models)
        device_label = "GPU" if "cuda" in str(self.device) else "CPU"
        self.statusBar().showMessage(
            f"  {mr_count} MR  •  {bt_count} BT modeli  •  {device_label}  •  Turan YZ"
        )

    # ── UI kurulumu ───────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background:{Colors.BG_PRIMARY};")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sayfa yığını
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background:{Colors.BG_PRIMARY};")
        main_layout.addWidget(self.stack, 1)

        # Start page
        self.start_page = StartPage()
        self.start_page.modality_selected.connect(self.show_mode_page)
        self.stack.addWidget(self.start_page)

    # ── Model yükleme ─────────────────────────────────────────────────────────
    def _load_models(self, path: str, model_cls, label: str) -> list:
        models = []
        if not os.path.isdir(path):
            print(f"Uyarı: '{path}' klasörü bulunamadı.")
            return models
        try:
            files = [f for f in os.listdir(path) if f.endswith(('.pt', '.pth'))]
            for fname in files:
                full = os.path.join(path, fname)
                model = model_cls()
                ckpt  = torch.load(full, map_location=self.device, weights_only=False)
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state, strict=False)
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"[OK] {full}")
            print(f"{len(models)} adet {label} modeli yüklendi.")
        except Exception as e:
            QMessageBox.critical(
                None, f"{label} Model Yükleme Hatası",
                f"Modeller yüklenirken hata oluştu:\n{e}"
            )
        return models

    # ── Sayfa geçişleri ───────────────────────────────────────────────────────
    def show_start_page(self):
        """Ana sayfaya dön, yığındaki tüm dinamik sayfaları temizle."""
        while self.stack.count() > 1:
            w = self.stack.widget(1)
            self.stack.removeWidget(w)
            w.deleteLater()
        self.page_history.clear()
        self.stack.setCurrentWidget(self.start_page)

    def show_mode_page(self, modality: str):
        self._push_current()
        mode_page = AnalysisModePage(modality)
        mode_page.mode_selected.connect(self.show_analysis_page)
        mode_page.back_clicked.connect(self.go_back)
        self.stack.addWidget(mode_page)
        self.stack.setCurrentWidget(mode_page)

    def show_analysis_page(self, modality: str, mode: str):
        self._push_current()

        models = self.mr_models if modality == "MR" else self.bt_models
        labels = self.label_names_mr if modality == "MR" else self.label_names_bt

        if not models:
            QMessageBox.warning(
                self, "Model Eksik",
                f"'{modality}' için yüklenmiş model bulunamadı."
            )
            # _push_current ile eklediğimizi geri al
            if self.page_history:
                self.page_history.pop()
            return

        if mode == "single":
            page = SingleAnalysisPage(modality, models, self.device, labels)
        elif modality == "MR":
            page = MultiAnalysisPageMR(modality, models, self.device, labels)
        else:
            page = MultiAnalysisPageBT(modality, models, self.device, labels)

        page.back_clicked.connect(self.go_back)
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def go_back(self):
        if self.page_history:
            current = self.stack.currentWidget()
            if hasattr(current, 'disconnect_worker_signals'):
                current.disconnect_worker_signals()

            prev = self.page_history.pop()

            # Önceki widget hâlâ stack'te mi kontrol et
            if self.stack.indexOf(prev) >= 0:
                self.stack.setCurrentWidget(prev)

            if current and self.stack.indexOf(current) >= 0:
                self.stack.removeWidget(current)
                current.deleteLater()
        else:
            self.show_start_page()

    def _push_current(self):
        current = self.stack.currentWidget()
        if current not in self.page_history:
            self.page_history.append(current)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroViva AI")
    app.setStyle("Fusion")

    # Fusion base palette
    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(Colors.BG_PRIMARY))
    palette.setColor(QPalette.WindowText,      QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Base,            QColor(Colors.BG_CARD))
    palette.setColor(QPalette.AlternateBase,   QColor(Colors.BG_CARD2))
    palette.setColor(QPalette.Text,            QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Button,          QColor(Colors.BG_SUBTLE))
    palette.setColor(QPalette.ButtonText,      QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight,       QColor(Colors.PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    window = MedicalImageAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
