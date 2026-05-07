# pages/single_analysis_page.py  —  YENİDEN TASARIMLANMIŞ
# Modern DICOM görüntü görüntüleyici + Dairesel gauge + Skeleton loader

import os
import numpy as np
import pydicom
import cv2
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QScrollArea, QFileDialog, QMessageBox, QSizePolicy,
    QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDragLeaveEvent, QDropEvent

from workers import AnalysisWorker
from ui.theme import Colors
from ui.history_manager import save_entry
from ui.custom_widgets import (
    CircularProgressWidget, ProbabilityBarWidget,
    SkeletonWidget, build_skeleton_result,
    DragDropZone, CriticalAlertWidget
)


# ─────────────────────────────────────────────────────────────────────────────
# Görüntü Görüntüleyici Araç Çubuğu
# ─────────────────────────────────────────────────────────────────────────────
class ViewerToolbar(QFrame):
    zoom_in_clicked   = pyqtSignal()
    zoom_out_clicked  = pyqtSignal()
    reset_clicked     = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_CARD};"
            f" border-bottom:1px solid {Colors.BORDER}; border-radius:0; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(6)

        for label, signal in [
            ("＋", self.zoom_in_clicked),
            ("－", self.zoom_out_clicked),
            ("⟳", self.reset_clicked),
        ]:
            btn = QPushButton(label)
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(
                f"QPushButton {{ background:{Colors.BG_SUBTLE}; color:{Colors.TEXT_SECONDARY};"
                f" border:1px solid {Colors.BORDER}; border-radius:4px; font-size:14px; }}"
                f"QPushButton:hover {{ color:{Colors.TEXT_PRIMARY}; border-color:{Colors.ACCENT}; }}"
            )
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(signal)
            layout.addWidget(btn)

        layout.addSpacing(10)

        # Pencereleme bilgisi (BT için WC/WW)
        self.wl_label = QLabel("WC: 40  WW: 80")
        self.wl_label.setStyleSheet(
            f"font-size:11px; color:{Colors.TEXT_MUTED};"
        )
        layout.addWidget(self.wl_label)

        layout.addStretch()

        # Görüntü boyut bilgisi
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            f"font-size:11px; color:{Colors.TEXT_MUTED};"
        )
        layout.addWidget(self.info_label)

    def set_info(self, text: str):
        self.info_label.setText(text)

    def set_wl(self, wc: int, ww: int):
        self.wl_label.setText(f"WC: {wc}  WW: {ww}")


# ─────────────────────────────────────────────────────────────────────────────
# DICOM / Görüntü Görüntüleyici Alanı
# ─────────────────────────────────────────────────────────────────────────────
class ImageViewerPanel(QFrame):
    def __init__(self, modality: str, parent=None):
        super().__init__(parent)
        self.modality = modality
        self._zoom = 1.0
        self._original_pixmap = None

        self.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_CARD}; border:1px solid {Colors.BORDER};"
            f" border-radius:8px; }}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.toolbar = ViewerToolbar(self)
        self.toolbar.zoom_in_clicked.connect(self._zoom_in)
        self.toolbar.zoom_out_clicked.connect(self._zoom_out)
        self.toolbar.reset_clicked.connect(self._zoom_reset)
        outer.addWidget(self.toolbar)

        # Görüntü alanı
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet(
            f"background:{Colors.BG_PRIMARY}; border:none;"
        )
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(220)
        self._show_placeholder()
        outer.addWidget(self.canvas, 1)

    def _show_placeholder(self):
        self.canvas.setText(
            "Görüntü yüklenmedi\n\nDosya yükledikten sonra\nburada görüntülenecek"
        )
        self.canvas.setStyleSheet(
            f"background:{Colors.BG_PRIMARY}; color:{Colors.TEXT_MUTED};"
            " font-size:12px; border:none;"
        )

    def load_image(self, file_path: str):
        try:
            array = None
            wc, ww = 40, 80

            if file_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(file_path)
                img = ds.pixel_array.astype(np.float32)

                if self.modality == 'BT':
                    intercept = float(getattr(ds, 'RescaleIntercept', 0))
                    slope     = float(getattr(ds, 'RescaleSlope', 1))
                    img = img * slope + intercept
                    wc  = int(getattr(ds, 'WindowCenter', 40))
                    ww  = int(getattr(ds, 'WindowWidth', 80))
                    if hasattr(wc, '__iter__'):
                        wc = int(wc[0])
                    if hasattr(ww, '__iter__'):
                        ww = int(ww[0])
                    lo, hi = wc - ww // 2, wc + ww // 2
                    img = np.clip(img, lo, hi)
                    self.toolbar.set_wl(wc, ww)
                else:
                    lo, hi = np.percentile(img, (1, 99))
                    img = np.clip(img, lo, hi)
                    self.toolbar.set_wl(0, 0)
                    self.toolbar.wl_label.setText("Min-Max normalize")

                span = hi - lo
                if span > 0:
                    img = (img - lo) / span
                array = (img * 255).astype(np.uint8)
                h, w = array.shape[:2]
                self.toolbar.set_info(f"{w}×{h}  |  Axial")
            else:
                array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if array is None:
                    raise ValueError("Görüntü okunamadı.")
                h, w = array.shape
                self.toolbar.set_info(f"{w}×{h}")

            pil = Image.fromarray(array)
            qimg = QImage(
                pil.tobytes(), pil.width, pil.height,
                pil.width, QImage.Format_Grayscale8
            )
            self._original_pixmap = QPixmap.fromImage(qimg)
            self._zoom = 1.0
            self._render()

        except Exception as e:
            self.canvas.setText(f"Görüntü yüklenemedi:\n{e}")
            self.canvas.setStyleSheet(
                f"background:{Colors.BG_PRIMARY}; color:{Colors.DANGER};"
                " font-size:11px; border:none;"
            )

    def _render(self):
        if self._original_pixmap is None:
            return
        available = self.canvas.size()
        scaled = self._original_pixmap.scaled(
            int(available.width()  * self._zoom),
            int(available.height() * self._zoom),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.canvas.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _zoom_in(self):
        self._zoom = min(self._zoom * 1.25, 4.0)
        self._render()

    def _zoom_out(self):
        self._zoom = max(self._zoom * 0.8, 0.25)
        self._render()

    def _zoom_reset(self):
        self._zoom = 1.0
        self._render()

    def clear(self):
        self._original_pixmap = None
        self._zoom = 1.0
        self._show_placeholder()
        self.toolbar.set_info("")
        self.toolbar.set_wl(40, 80)


# ─────────────────────────────────────────────────────────────────────────────
# Sonuç Paneli
# ─────────────────────────────────────────────────────────────────────────────
class ResultPanel(QFrame):
    """Analiz sonuçlarını dairesel gauge + progress bar ile gösteren panel."""

    def __init__(self, label_names: list, parent=None):
        super().__init__(parent)
        self.label_names = label_names
        self.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_CARD}; border:1px solid {Colors.BORDER};"
            f" border-radius:8px; }}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(14, 12, 14, 12)
        self._outer.setSpacing(10)

        # Başlık
        header_row = QHBoxLayout()
        title = QLabel("Analiz Sonuçları")
        title.setStyleSheet(
            f"font-size:15px; font-weight:600; color:{Colors.TEXT_PRIMARY};"
        )
        self._status_badge = QLabel("  Bekleniyor  ")
        self._status_badge.setStyleSheet(
            f"background:{Colors.BG_SUBTLE}; color:{Colors.TEXT_SECONDARY};"
            " border-radius:4px; font-size:11px; font-weight:500;"
        )
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self._status_badge)
        self._outer.addLayout(header_row)

        # İçerik alanı (scroll)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("border:none; background:transparent;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._content = QWidget()
        self._content.setStyleSheet("background:transparent;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._scroll.setWidget(self._content)
        self._outer.addWidget(self._scroll, 1)

        self.show_idle()

    # ── İçerik temizle ───────────────────────────────────────────────────────
    def _clear_content(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    # ── Durumlar ─────────────────────────────────────────────────────────────
    def show_idle(self):
        self._clear_content()
        self._set_badge("Bekleniyor", Colors.TEXT_SECONDARY, Colors.BG_SUBTLE)
        msg = QLabel("Analiz için dosya yükleyin")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(
            f"font-size:12px; color:{Colors.TEXT_MUTED}; padding:40px 0;"
        )
        self._content_layout.addWidget(msg)
        self._content_layout.addStretch()

    def show_loading(self, message: str = "Yapay Zeka Analiz Ediyor..."):
        self._clear_content()
        self._set_badge("Analiz Ediliyor", Colors.PRIMARY, Colors.DANGER_BG)

        lbl = QLabel(message)
        lbl.setStyleSheet(
            f"font-size:13px; font-weight:600; color:{Colors.PRIMARY};"
        )
        self._content_layout.addWidget(lbl)

        sub = QLabel("Model görüntüyü işliyor, lütfen bekleyin.")
        sub.setStyleSheet(f"font-size:11px; color:{Colors.TEXT_SECONDARY};")
        self._content_layout.addWidget(sub)
        self._content_layout.addSpacing(10)

        for h in [16, 10, 10, 70, 10, 10, 10]:
            sk = SkeletonWidget(width=320, height=h, radius=5 if h > 20 else 3)
            self._content_layout.addWidget(sk)
        self._content_layout.addStretch()

    def update_loading_text(self, message: str):
        # İlk widget loading label ise güncelle
        item = self._content_layout.itemAt(0)
        if item and item.widget():
            w = item.widget()
            if isinstance(w, QLabel):
                w.setText(message)

    def show_results(self, prediction: str, probabilities: list):
        self._clear_content()
        self._set_badge("Tamamlandı", "#66BB6A", Colors.SUCCESS_BG)

        # ── Tespit edilen durum kutusu ────────────────────────────────────────
        pred_frame = QFrame()
        pred_frame.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_CARD2}; border:1px solid {Colors.BORDER};"
            f" border-radius:8px; }}"
        )
        pred_layout = QVBoxLayout(pred_frame)
        pred_layout.setContentsMargins(12, 10, 12, 10)
        pred_layout.setSpacing(3)

        pl = QLabel("TESPİT EDİLEN DURUM")
        pl.setStyleSheet(
            f"font-size:11px; color:{Colors.TEXT_MUTED}; letter-spacing:1px; font-weight:600;"
        )
        pred_layout.addWidget(pl)

        # İnme tespiti ise kırmızı, değilse yeşil
        is_critical = any(
            crit in prediction
            for crit in ("İnme", "HiperakutAkut", "Subakut")
        )
        pred_color = Colors.DANGER if is_critical else Colors.SUCCESS

        pv = QLabel(prediction)
        pv.setStyleSheet(
            f"font-size:20px; font-weight:700; color:{pred_color};"
        )
        pred_layout.addWidget(pv)

        ensemble_lbl = QLabel(f"Ensemble ortalama — {len(probabilities)} sınıf")
        ensemble_lbl.setStyleSheet(
            f"font-size:11px; color:{Colors.TEXT_MUTED};"
        )
        pred_layout.addWidget(ensemble_lbl)
        self._content_layout.addWidget(pred_frame)

        # ── Dairesel gauge'lar ────────────────────────────────────────────────
        gauge_colors = self._get_gauge_colors(prediction)
        gauges_row = QHBoxLayout()
        gauges_row.setSpacing(8)

        for i, (lname, prob) in enumerate(
            zip(self.label_names, probabilities)
        ):
            color = gauge_colors.get(lname, Colors.TEXT_SECONDARY)
            gauge_wrap = QFrame()
            gauge_wrap.setStyleSheet(
                f"QFrame {{ background:{Colors.BG_CARD2}; border:1px solid {Colors.BORDER};"
                f" border-radius:8px; }}"
            )
            gw_layout = QVBoxLayout(gauge_wrap)
            gw_layout.setContentsMargins(6, 8, 6, 8)
            gw_layout.setAlignment(Qt.AlignCenter)

            gauge = CircularProgressWidget(
                value=prob, color=color,
                label=lname, size=100
            )
            gw_layout.addWidget(gauge, alignment=Qt.AlignCenter)
            gauges_row.addWidget(gauge_wrap)

        self._content_layout.addLayout(gauges_row)

        # ── Olasılık çubukları ────────────────────────────────────────────────
        bar_title = QLabel("Olasılık Dağılımı")
        bar_title.setStyleSheet(
            f"font-size:13px; font-weight:600; color:{Colors.TEXT_SECONDARY};"
        )
        self._content_layout.addWidget(bar_title)

        bar_frame = QFrame()
        bar_frame.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_CARD2}; border:1px solid {Colors.BORDER};"
            f" border-radius:8px; }}"
        )
        bar_layout = QVBoxLayout(bar_frame)
        bar_layout.setContentsMargins(12, 10, 12, 10)
        bar_layout.setSpacing(8)

        for lname, prob in zip(self.label_names, probabilities):
            color     = gauge_colors.get(lname, Colors.TEXT_SECONDARY)
            highlight = lname in prediction
            bar = ProbabilityBarWidget(lname, prob, color=color, highlight=highlight)
            bar_layout.addWidget(bar)

        self._content_layout.addWidget(bar_frame)

        # ── Kritik uyarı kutusu (yalnızca inme/patoloji tespitinde) ──────────
        if is_critical:
            max_prob = max(probabilities)
            alert = CriticalAlertWidget(
                title="Kritik Bulgu",
                message=(
                    f"Yüksek patoloji olasılığı (%{max_prob:.1f}) tespit edildi. "
                    "Lütfen radyolog değerlendirmesine başvurun."
                ),
                level="danger"
            )
            self._content_layout.addWidget(alert)

        self._content_layout.addStretch()

    def show_error(self, error_message: str):
        self._clear_content()
        self._set_badge("Hata", Colors.DANGER, Colors.DANGER_BG)

        alert = CriticalAlertWidget(
            title="Analiz Hatası",
            message=error_message,
            level="danger"
        )
        self._content_layout.addWidget(alert)
        self._content_layout.addStretch()

    # ── Yardımcı metotlar ─────────────────────────────────────────────────────
    def _set_badge(self, text: str, color: str, bg: str):
        self._status_badge.setText(f"  {text}  ")
        self._status_badge.setStyleSheet(
            f"background:{bg}; color:{color};"
            " border-radius:4px; font-size:10px; font-weight:500;"
        )

    def _get_gauge_colors(self, prediction: str) -> dict:
        """Her sınıf etiketi için renk döndür."""
        colors = {}
        for lname in self.label_names:
            if lname in prediction:
                if lname in ("İnme", "HiperakutAkut"):
                    colors[lname] = Colors.DANGER
                elif lname == "Subakut":
                    colors[lname] = Colors.WARNING
                else:
                    colors[lname] = Colors.SUCCESS
            else:
                if lname in ("İnme", "HiperakutAkut"):
                    colors[lname] = "#7B3333"
                elif lname == "Subakut":
                    colors[lname] = "#7B5A1A"
                else:
                    colors[lname] = "#2A5A2E"
        return colors


# ─────────────────────────────────────────────────────────────────────────────
# Sol Panel — Dosya Yükleme + Görüntü Görüntüleyici
# ─────────────────────────────────────────────────────────────────────────────
class LeftPanel(QFrame):
    file_selected = pyqtSignal(str)

    def __init__(self, modality: str, parent=None):
        super().__init__(parent)
        self.modality  = modality
        self._current_file = None

        self.setStyleSheet("QFrame { background:transparent; border:none; }")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Sürükle-bırak bölgesi ─────────────────────────────────────────────
        self.drop_zone = DragDropZone(self)
        self.drop_zone.file_dropped.connect(self._on_file)
        layout.addWidget(self.drop_zone)

        # ── Dosya seç butonu ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_select = QPushButton("Dosya Seç")
        self.btn_select.setObjectName("btn_primary")
        self.btn_select.setFixedHeight(36)
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.clicked.connect(self._open_dialog)

        self.btn_clear = QPushButton("Temizle")
        self.btn_clear.setFixedHeight(36)
        self.btn_clear.setFixedWidth(80)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self._clear)

        btn_row.addWidget(self.btn_select, 1)
        btn_row.addWidget(self.btn_clear)
        layout.addLayout(btn_row)

        # ── Görüntü görüntüleyici ─────────────────────────────────────────────
        self.viewer = ImageViewerPanel(modality, self)
        layout.addWidget(self.viewer, 1)

    def _open_dialog(self):
        fmt = "Desteklenen Görüntüler (*.dcm *.png *.jpg *.jpeg)"
        path, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "", fmt)
        if path:
            self._on_file(path)

    def _on_file(self, path: str):
        self._current_file = path
        fname = os.path.basename(path)
        self.drop_zone.set_file(fname)
        self.viewer.load_image(path)
        self.file_selected.emit(path)

    def _clear(self):
        self._current_file = None
        self.drop_zone.clear()
        self.viewer.clear()

    # ── Drag & Drop: tüm widget'a yönlendir ──────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        self.drop_zone.dragEnterEvent(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.drop_zone.dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self.drop_zone.dropEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Ana Sayfa Sınıfı
# ─────────────────────────────────────────────────────────────────────────────
class SingleAnalysisPage(QWidget):
    back_clicked = pyqtSignal()

    def __init__(self, modality: str, models: list, device, label_names: list,
                 parent=None):
        super().__init__(parent)
        self.modality    = modality
        self.models      = models
        self.device      = device
        self.label_names = label_names
        self._worker     = None
        self._current_file = None
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Üst bar ────────────────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_PRIMARY};"
            f" border-bottom:1px solid {Colors.BORDER}; }}"
        )
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)

        back_btn = QPushButton("← Geri")
        back_btn.setObjectName("btn_back")
        back_btn.setFixedHeight(32)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.back_clicked)

        mod_color = Colors.PRIMARY if self.modality == "BT" else Colors.ACCENT
        mod_bg = Colors.BT_BG if self.modality == "BT" else Colors.MR_BG

        title_lbl = QLabel(f"{self.modality}  •  Tekli Analiz")
        title_lbl.setStyleSheet(
            f"font-size:16px; font-weight:600; color:{Colors.TEXT_PRIMARY};"
        )

        badge_mod = QLabel(f"  {self.modality}  ")
        badge_mod.setStyleSheet(
            f"background:{mod_bg}; color:{mod_color}; border-radius:4px;"
            " font-size:11px; font-weight:600;"
        )

        tb.addWidget(back_btn)
        tb.addSpacing(16)
        tb.addWidget(title_lbl)
        tb.addStretch()
        tb.addWidget(badge_mod)
        root.addWidget(topbar)

        # ── İçerik alanı ─────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet(f"background:{Colors.BG_PRIMARY};")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 12)
        content_layout.setSpacing(12)

        # Sol panel
        self.left_panel = LeftPanel(self.modality, self)
        self.left_panel.file_selected.connect(self._on_file_selected)
        content_layout.addWidget(self.left_panel)

        # Sağ panel — Sonuçlar
        self.result_panel = ResultPanel(self.label_names, self)
        content_layout.addWidget(self.result_panel, 1)

        root.addWidget(content, 1)

    # ── Sürükle-bırak (tüm sayfa seviyesinde) ────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            path = event.mimeData().urls()[0].toLocalFile()
            supported = ('.dcm', '.png', '.jpg', '.jpeg')
            if path.lower().endswith(supported):
                self.left_panel._on_file(path)
            else:
                QMessageBox.warning(
                    self, "Desteklenmeyen Dosya",
                    "Lütfen .dcm, .png, .jpg veya .jpeg formatında dosya seçin."
                )

    # ── Analiz iş akışı ───────────────────────────────────────────────────────
    def _on_file_selected(self, path: str):
        self._current_file = path
        self._run_analysis(path)

    def _run_analysis(self, file_path: str):
        # Önceki worker varsa durdur
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(500)

        self.result_panel.show_loading()

        self._worker = AnalysisWorker(
            self.models, self.device, file_path,
            self.label_names, self.modality
        )
        self._worker.progress.connect(self.result_panel.update_loading_text)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, prediction: str, probabilities: list):
        self.result_panel.show_results(prediction, probabilities)
        # Geçmişe kaydet
        try:
            import os
            fname = os.path.basename(self._current_file) if self._current_file else "bilinmeyen"
            max_prob = max(probabilities) if probabilities else 0
            save_entry({
                "filename": fname,
                "modality": self.modality,
                "result": prediction,
                "confidence": f"{max_prob:.1f}",
            })
        except Exception:
            pass

    def _on_error(self, error_message: str):
        self.result_panel.show_error(error_message)

    def disconnect_worker_signals(self):
        """main.py go_back() çağrısı için güvenli sinyal kesme."""
        if self._worker:
            try:
                self._worker.progress.disconnect()
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except Exception:
                pass
