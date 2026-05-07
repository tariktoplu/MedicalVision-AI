# pages/analysis_mode_page.py
# Analiz modu seçim sayfası — Tekli / Çoklu
# Büyük, ikonlu, profesyonel kart tasarımı

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPainterPath

from ui.theme import Colors


# ─────────────────────────────────────────────────────────────────────────────
# SVG-style ikonlar QPainter ile
# ─────────────────────────────────────────────────────────────────────────────
class FileIconWidget(QWidget):
    """Tek dosya ikonu — QPainter ile çizilir."""
    def __init__(self, color: str, size: int = 40, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.width()
        m = s * 0.15  # margin

        # Dosya gövdesi
        pen = QPen(self._color, 1.8)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Ana dikdörtgen (sol alt → sağ üst, katlı köşe)
        x1, y1 = m, m
        x2, y2 = s - m, s - m
        fold = s * 0.25  # katlanma boyutu

        path = QPainterPath()
        path.moveTo(x1, y1)
        path.lineTo(x2 - fold, y1)
        path.lineTo(x2, y1 + fold)
        path.lineTo(x2, y2)
        path.lineTo(x1, y2)
        path.closeSubpath()
        p.drawPath(path)

        # Katlama çizgisi
        p.drawLine(int(x2 - fold), int(y1), int(x2 - fold), int(y1 + fold))
        p.drawLine(int(x2 - fold), int(y1 + fold), int(x2), int(y1 + fold))

        # Satır çizgileri (içerik gösterimi)
        line_y_start = y1 + fold + s * 0.15
        line_x1 = x1 + s * 0.12
        line_x2_long = x2 - s * 0.12
        line_x2_short = x1 + s * 0.45
        gap = s * 0.12

        p.setPen(QPen(self._color, 1.2))
        for i, short in enumerate([False, True, False]):
            ly = line_y_start + i * gap
            if ly < y2 - s * 0.08:
                lx2 = line_x2_short if short else line_x2_long
                p.drawLine(int(line_x1), int(ly), int(lx2), int(ly))

        p.end()


class MultiFileIconWidget(QWidget):
    """Çoklu dosya ikonu — üst üste binmiş dosyalar."""
    def __init__(self, color: str, size: int = 40, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.width()

        pen_bg = QPen(QColor(self._color.red(), self._color.green(), self._color.blue(), 80), 1.4)
        pen_fg = QPen(self._color, 1.8)

        # Arka dosya (sağ üste kaydırılmış)
        off = s * 0.1
        m = s * 0.15
        x1, y1 = m + off, m - off * 0.3
        x2, y2 = s - m + off, s - m - off * 0.8
        fold = s * 0.2

        p.setPen(pen_bg)
        path1 = QPainterPath()
        path1.moveTo(x1, y1)
        path1.lineTo(x2 - fold, y1)
        path1.lineTo(x2, y1 + fold)
        path1.lineTo(x2, y2)
        path1.lineTo(x1, y2)
        path1.closeSubpath()
        p.drawPath(path1)

        # Ön dosya
        x1, y1 = m - off * 0.3, m + off * 0.5
        x2, y2 = s - m - off * 0.3, s - m
        fold = s * 0.22

        p.setPen(pen_fg)
        path2 = QPainterPath()
        path2.moveTo(x1, y1)
        path2.lineTo(x2 - fold, y1)
        path2.lineTo(x2, y1 + fold)
        path2.lineTo(x2, y2)
        path2.lineTo(x1, y2)
        path2.closeSubpath()
        p.drawPath(path2)

        p.drawLine(int(x2 - fold), int(y1), int(x2 - fold), int(y1 + fold))
        p.drawLine(int(x2 - fold), int(y1 + fold), int(x2), int(y1 + fold))

        # Satır çizgileri
        line_y_start = y1 + fold + s * 0.12
        line_x1 = x1 + s * 0.1
        line_x2 = x2 - s * 0.1
        gap = s * 0.11

        p.setPen(QPen(self._color, 1.2))
        for i in range(3):
            ly = line_y_start + i * gap
            lx2 = line_x1 + (line_x2 - line_x1) * (0.6 if i == 1 else 0.85)
            if ly < y2 - s * 0.06:
                p.drawLine(int(line_x1), int(ly), int(lx2), int(ly))

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Analiz Modu Kartı — Büyük, profesyonel
# ─────────────────────────────────────────────────────────────────────────────
class ModeCard(QFrame):
    """Tekli / Çoklu analiz seçim kartı — büyük ikon + açıklama."""
    clicked = pyqtSignal(str)

    def __init__(self, mode: str, title: str, description: str,
                 features: list, accent: str, accent_bg: str,
                 icon_widget: QWidget, parent=None):
        super().__init__(parent)
        self._mode = mode
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(240)
        self.setObjectName("modeCard")
        self.setStyleSheet(
            f"QFrame#modeCard {{ background:{Colors.BG_CARD}; border:1px solid {Colors.BORDER};"
            f" border-radius:12px; }}"
            f"QFrame#modeCard:hover {{ border-color:{accent}; background:{Colors.BG_CARD2}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(0)

        # ── İkon alanı ────────────────────────────────────────────────────────
        icon_frame = QFrame()
        icon_frame.setFixedSize(64, 64)
        icon_frame.setStyleSheet(
            f"background:{accent_bg}; border-radius:14px; border:none;"
        )
        icon_inner = QHBoxLayout(icon_frame)
        icon_inner.setContentsMargins(0, 0, 0, 0)
        icon_inner.addWidget(icon_widget, alignment=Qt.AlignCenter)
        layout.addWidget(icon_frame)
        layout.addSpacing(18)

        # ── Başlık ────────────────────────────────────────────────────────────
        t = QLabel(title)
        t.setStyleSheet(
            f"font-size:20px; font-weight:700; color:{Colors.TEXT_PRIMARY};"
        )
        layout.addWidget(t)
        layout.addSpacing(6)

        # ── Açıklama ─────────────────────────────────────────────────────────
        d = QLabel(description)
        d.setWordWrap(True)
        d.setStyleSheet(f"font-size:14px; color:{Colors.TEXT_SECONDARY}; line-height:1.4;")
        layout.addWidget(d)
        layout.addSpacing(16)

        # ── Özellik listesi ──────────────────────────────────────────────────
        for feat in features:
            feat_row = QHBoxLayout()
            feat_row.setSpacing(8)
            feat_row.setContentsMargins(0, 0, 0, 0)

            dot = QLabel("•")
            dot.setFixedWidth(12)
            dot.setStyleSheet(f"font-size:14px; color:{accent};")
            feat_row.addWidget(dot)

            feat_lbl = QLabel(feat)
            feat_lbl.setStyleSheet(f"font-size:13px; color:{Colors.TEXT_SECONDARY};")
            feat_row.addWidget(feat_lbl)
            feat_row.addStretch()

            layout.addLayout(feat_row)
            layout.addSpacing(2)

        layout.addStretch()

        # ── Alt buton alanı ──────────────────────────────────────────────────
        btn = QPushButton(f"{title} →")
        btn.setFixedHeight(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background:{accent_bg}; color:{accent};"
            f" border:1px solid {accent}; border-radius:8px;"
            f" font-size:14px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{accent}; color:{Colors.BG_PRIMARY}; }}"
        )
        btn.clicked.connect(lambda: self.clicked.emit(self._mode))
        layout.addWidget(btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._mode)
        super().mousePressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
class AnalysisModePage(QWidget):
    """Analiz modu seçim sayfası — Tekli / Çoklu."""
    mode_selected = pyqtSignal(str, str)
    back_clicked  = pyqtSignal()

    def __init__(self, modality: str, parent=None):
        super().__init__(parent)
        self.modality = modality
        self._accent = Colors.PRIMARY if modality == "BT" else Colors.ACCENT
        self._accent_bg = Colors.BT_BG if modality == "BT" else Colors.MR_BG
        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Üst bar ──────────────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setFixedHeight(56)
        topbar.setStyleSheet(
            f"QFrame {{ background:{Colors.BG_PRIMARY};"
            f" border-bottom:1px solid {Colors.BORDER}; }}"
        )
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(28, 0, 28, 0)

        back_btn = QPushButton("← Geri")
        back_btn.setObjectName("btn_back")
        back_btn.setFixedHeight(34)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.back_clicked)

        page_title = QLabel(f"{self.modality} Analizi")
        page_title.setStyleSheet(
            f"font-size:18px; font-weight:600; color:{Colors.TEXT_PRIMARY};"
        )

        mod_badge = QLabel(f"  {self.modality}  ")
        mod_badge.setStyleSheet(
            f"background:{self._accent_bg}; color:{self._accent};"
            f" border-radius:4px; font-size:12px; font-weight:600;"
            f" padding:2px 6px;"
        )

        tb.addWidget(back_btn)
        tb.addSpacing(18)
        tb.addWidget(page_title)
        tb.addStretch()
        tb.addWidget(mod_badge)
        root.addWidget(topbar)

        # ── İçerik ────────────────────────────────────────────────────────────
        center = QWidget()
        center.setStyleSheet(f"background:{Colors.BG_PRIMARY};")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(40, 32, 40, 32)
        center_layout.setSpacing(0)

        # Başlık
        heading = QLabel("Analiz türünü seçin")
        heading.setStyleSheet(
            f"font-size:24px; font-weight:700; color:{Colors.TEXT_PRIMARY};"
        )
        center_layout.addWidget(heading)
        center_layout.addSpacing(6)

        sub = QLabel(f"{self.modality} modalitesi için tek veya toplu analiz yapabilirsiniz.")
        sub.setStyleSheet(f"font-size:14px; color:{Colors.TEXT_SECONDARY};")
        center_layout.addWidget(sub)
        center_layout.addSpacing(28)

        # ── Kartlar yan yana ─────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(20)

        # Tekli Analiz kartı
        single_icon = FileIconWidget(self._accent, size=36)
        single_card = ModeCard(
            mode="single",
            title="Tekli Analiz",
            description="Tek bir DICOM veya görüntü dosyası yükleyerek anlık AI analizi yapın.",
            features=[
                "Tek dosya yükleme",
                "Anlık sonuç ve olasılık dağılımı",
                "Dairesel gauge gösterimi",
                "Görüntü görüntüleyici ile detaylı inceleme",
            ],
            accent=self._accent,
            accent_bg=self._accent_bg,
            icon_widget=single_icon,
        )
        single_card.clicked.connect(
            lambda m: self.mode_selected.emit(self.modality, m)
        )
        cards_row.addWidget(single_card)

        # Çoklu Analiz kartı
        multi_icon = MultiFileIconWidget(Colors.SUCCESS, size=36)
        multi_card = ModeCard(
            mode="multi",
            title="Çoklu Analiz",
            description="Klasör veya birden fazla dosya yükleyerek toplu analiz gerçekleştirin.",
            features=[
                "Klasör veya çoklu dosya seçimi",
                "Toplu tahmin ve ilerleme takibi",
                "Sonuçları JSON formatında dışa aktarma",
                "Analiz sonuç özeti ve istatistikler",
            ],
            accent=Colors.SUCCESS,
            accent_bg=Colors.SUCCESS_BG,
            icon_widget=multi_icon,
        )
        multi_card.clicked.connect(
            lambda m: self.mode_selected.emit(self.modality, m)
        )
        cards_row.addWidget(multi_card)

        center_layout.addLayout(cards_row, 1)
        center_layout.addSpacing(20)

        # Desteklenen format bilgisi
        info_row = QHBoxLayout()
        info_row.setSpacing(6)

        info_label = QLabel("Desteklenen formatlar:")
        info_label.setStyleSheet(f"font-size:12px; color:{Colors.TEXT_MUTED};")
        info_row.addWidget(info_label)

        for fmt in [".dcm", ".png", ".jpg", ".jpeg"]:
            fmt_badge = QLabel(f" {fmt} ")
            fmt_badge.setStyleSheet(
                f"background:{Colors.BG_CARD}; color:{Colors.TEXT_SECONDARY};"
                f" border-radius:4px; font-size:11px; font-weight:500;"
                f" padding:2px 6px; border:1px solid {Colors.BORDER};"
            )
            info_row.addWidget(fmt_badge)

        info_row.addStretch()
        center_layout.addLayout(info_row)

        root.addWidget(center, 1)
