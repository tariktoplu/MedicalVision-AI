# ui/custom_widgets.py
# Özel PyQt5 widget sınıfları — NeuroViva-AI Minimalist UI Kütüphanesi
#
# İçerik:
#   - CircularProgressWidget : Dairesel gauge chart (QPainter)
#   - ProbabilityBarWidget   : Renkli progress bar + etiket
#   - SkeletonWidget         : Yükleniyor animasyonu (QPropertyAnimation)
#   - DragDropZone           : Gelişmiş sürükle-bırak alanı
#   - ResultCard             : Sonuç özet kartı
#   - CriticalAlertWidget    : Kritik bulgu uyarısı

import math
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QFrame, QGraphicsOpacityEffect)
from PyQt5.QtCore import (Qt, QTimer, QPropertyAnimation, QEasingCurve,
                          QRectF, pyqtProperty, QSequentialAnimationGroup,
                          pyqtSignal)
from PyQt5.QtGui import (QPainter, QColor, QPen, QFont, QConicalGradient,
                         QLinearGradient, QDragEnterEvent, QDragLeaveEvent,
                         QDropEvent, QFontMetrics)

from .theme import Colors


# ─────────────────────────────────────────────────────────────────────────────
# 1. CircularProgressWidget — QPainter ile dairesel gauge
# ─────────────────────────────────────────────────────────────────────────────
class CircularProgressWidget(QWidget):
    """
    Dairesel ilerleme/olasılık göstergesi.
    Kullanım:
        w = CircularProgressWidget(value=79.4, color="#E63946", label="İnme")
        layout.addWidget(w)
    """

    def __init__(self, value: float = 0.0, color: str = Colors.PRIMARY,
                 label: str = "", size: int = 90, parent=None):
        super().__init__(parent)
        self._value   = 0.0          # animasyon için başlangıç
        self._target  = float(value)
        self._color   = QColor(color)
        self._label   = label
        self._size    = size
        self.setFixedSize(size, size + 20)  # +20 alt etiket için

        # Animasyonlu değer artışı
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)    # ~60 fps
        self._anim_timer.timeout.connect(self._step)
        self._anim_timer.start()

    def set_value(self, value: float, color: str = None):
        self._target = float(value)
        if color:
            self._color = QColor(color)
        if not self._anim_timer.isActive():
            self._anim_timer.start()

    def _step(self):
        diff = self._target - self._value
        if abs(diff) < 0.5:
            self._value = self._target
            self._anim_timer.stop()
        else:
            self._value += diff * 0.12
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        margin  = 8
        w = self._size
        arc_rect = QRectF(margin, margin, w - 2 * margin, w - 2 * margin)
        stroke   = 5  # daha ince çizgi — minimalist
        span     = int((self._value / 100.0) * 360 * 16)
        start    = 90 * 16

        # Arka iz (track) — çok ince
        pen_track = QPen(QColor(Colors.BG_SUBTLE), stroke, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_track)
        painter.drawArc(arc_rect, 0, 360 * 16)

        # Değer yayı
        if span > 0:
            pen_arc = QPen(self._color, stroke, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen_arc)
            painter.drawArc(arc_rect, start, -span)

        # Merkez metin — yüzde
        painter.setPen(QPen(QColor(Colors.TEXT_PRIMARY)))
        pct_font = QFont("Segoe UI", max(8, self._size // 9), QFont.DemiBold)
        painter.setFont(pct_font)
        painter.drawText(
            QRectF(0, 4, w, w - 4),
            Qt.AlignCenter,
            f"%{self._value:.0f}"
        )

        # Alt etiket
        if self._label:
            painter.setPen(QPen(QColor(Colors.TEXT_MUTED)))
            lbl_font = QFont("Segoe UI", max(7, self._size // 12))
            painter.setFont(lbl_font)
            painter.drawText(
                QRectF(0, w, w, 20),
                Qt.AlignCenter,
                self._label
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ProbabilityBarWidget — Renkli bar + etiket + yüzde
# ─────────────────────────────────────────────────────────────────────────────
class ProbabilityBarWidget(QWidget):
    """
    Tek bir olasılık satırı.
    Kullanım:
        bar = ProbabilityBarWidget("İnme", 79.4, color="#E63946", highlight=True)
    """

    def __init__(self, label: str, value: float, color: str = Colors.PRIMARY,
                 highlight: bool = False, parent=None):
        super().__init__(parent)
        self._target = float(value)
        self._color  = color

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(3)

        # Etiket + yüzde satırı
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"font-size:13px; color:{'%s' % color if highlight else Colors.TEXT_SECONDARY};"
            f"font-weight:{'600' if highlight else '400'};"
        )

        pct = QLabel(f"%{value:.1f}")
        pct.setStyleSheet(
            f"font-size:13px; font-weight:600; color:{color};"
        )

        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(pct)
        outer.addLayout(row)

        # Bar track
        track = QFrame()
        track.setFixedHeight(4)
        track.setStyleSheet(
            f"background:{Colors.BG_SUBTLE}; border-radius:2px;"
        )
        track_layout = QHBoxLayout(track)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.setSpacing(0)

        self._fill = QFrame(track)
        self._fill.setFixedHeight(4)
        self._fill.setStyleSheet(
            f"background:{color}; border-radius:2px;"
        )
        # Animasyonlu genişlik
        self._fill.setFixedWidth(0)
        outer.addWidget(track)

        # Animasyon — genişliği artır
        self._track = track
        QTimer.singleShot(100, self._animate)

    def _animate(self):
        w = int(self._track.width() * self._target / 100)
        anim = QPropertyAnimation(self._fill, b"minimumWidth")
        anim.setDuration(600)
        anim.setStartValue(0)
        anim.setEndValue(w)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._anim_ref = anim  # referansı koru


# ─────────────────────────────────────────────────────────────────────────────
# 3. SkeletonWidget — Yükleniyor iskelet animasyonu
# ─────────────────────────────────────────────────────────────────────────────
class SkeletonWidget(QFrame):
    """
    Yarı saydam titreşen gri yükleme iskeleti.
    Kullanım:
        skeleton = SkeletonWidget(width=300, height=20)
    """

    def __init__(self, width: int = 200, height: int = 16, radius: int = 5,
                 parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setStyleSheet(
            f"background:{Colors.BG_CARD2}; border-radius:{radius}px;"
        )
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(900)
        anim.setStartValue(0.3)
        anim.setEndValue(0.8)
        anim.setEasingCurve(QEasingCurve.InOutSine)

        group = QSequentialAnimationGroup(self)
        group.addAnimation(anim)
        rev = QPropertyAnimation(effect, b"opacity")
        rev.setDuration(900)
        rev.setStartValue(0.8)
        rev.setEndValue(0.3)
        rev.setEasingCurve(QEasingCurve.InOutSine)
        group.addAnimation(rev)
        group.setLoopCount(-1)  # sonsuz
        group.start()
        self._group = group


def build_skeleton_result() -> QWidget:
    """
    Analiz süresince gösterilecek skeleton loader seti.
    show_results() çağrıldığında bu widget kaldırılır.
    """
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setSpacing(12)
    layout.setContentsMargins(4, 4, 4, 4)

    ai_label = QLabel("Yapay Zeka Analiz Ediyor...")
    ai_label.setStyleSheet(
        f"font-size:13px; font-weight:600; color:{Colors.PRIMARY};"
    )
    layout.addWidget(ai_label)

    sub = QLabel("Model görüntüyü işliyor, lütfen bekleyin.")
    sub.setStyleSheet(f"font-size:11px; color:{Colors.TEXT_SECONDARY};")
    layout.addWidget(sub)
    layout.addSpacing(8)

    for h in [16, 12, 12, 60, 12, 12]:
        sk = SkeletonWidget(width=260, height=h, radius=5 if h > 20 else 3)
        layout.addWidget(sk)
        if h == 60:
            layout.addSpacing(4)

    layout.addStretch()
    return container


# ─────────────────────────────────────────────────────────────────────────────
# 4. DragDropZone — Gelişmiş sürükle-bırak alanı
# ─────────────────────────────────────────────────────────────────────────────
class DragDropZone(QLabel):
    """
    Dosya sürükle-bırak bölgesi.
    Sinyaller: file_dropped(str)
    """
    file_dropped = pyqtSignal(str)

    SUPPORTED = ('.dcm', '.png', '.jpg', '.jpeg')

    _STYLE_IDLE = (
        f"QLabel {{ background:{Colors.BG_CARD}; border:1px dashed {Colors.BORDER_MED};"
        f" border-radius:10px; color:{Colors.TEXT_MUTED}; font-size:13px; }}"
    )
    _STYLE_HOVER = (
        f"QLabel {{ background:{Colors.BG_CARD2}; border:1px dashed {Colors.ACCENT};"
        f" border-radius:10px; color:{Colors.TEXT_PRIMARY}; font-size:13px; }}"
    )
    _STYLE_LOADED = (
        f"QLabel {{ background:{Colors.SUCCESS_BG}; border:1px solid {Colors.SUCCESS};"
        f" border-radius:10px; color:{Colors.SUCCESS}; font-size:13px; }}"
    )
    _STYLE_ERROR = (
        f"QLabel {{ background:{Colors.DANGER_BG}; border:1px solid {Colors.DANGER};"
        f" border-radius:10px; color:{Colors.DANGER}; font-size:13px; }}"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(140)
        self.setWordWrap(True)
        self._reset_text()
        self.setStyleSheet(self._STYLE_IDLE)

    def _reset_text(self):
        self.setText(
            "Dosyayı buraya sürükleyin\n"
            "veya 'Dosya Seç' butonuna tıklayın\n\n"
            ".dcm  .png  .jpg  .jpeg"
        )

    def set_file(self, filename: str):
        self.setText(f"✓  {filename}")
        self.setStyleSheet(self._STYLE_LOADED)

    def set_error(self, msg: str = "Desteklenmeyen format"):
        self.setText(f"✗  {msg}")
        self.setStyleSheet(self._STYLE_ERROR)
        QTimer.singleShot(2000, lambda: (
            self._reset_text(),
            self.setStyleSheet(self._STYLE_IDLE)
        ))

    def clear(self):
        self._reset_text()
        self.setStyleSheet(self._STYLE_IDLE)

    # ── Drag & Drop olayları ─────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()
            self.setStyleSheet(self._STYLE_HOVER)
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.setStyleSheet(self._STYLE_IDLE)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            path = event.mimeData().urls()[0].toLocalFile()
            if path.lower().endswith(self.SUPPORTED):
                self.file_dropped.emit(path)
            else:
                self.set_error("Desteklenmeyen dosya formatı")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ResultCard — Metrik kart (Dashboard istatistikleri)
# ─────────────────────────────────────────────────────────────────────────────
class ResultCard(QFrame):
    """
    Basit metrik özet kartı.
    Kullanım:
        card = ResultCard(title="Toplam Analiz", value="2,847",
                          subtitle="+12 bu hafta")
    """

    def __init__(self, title: str, value: str, subtitle: str = "",
                 value_color: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(3)

        t = QLabel(title.upper())
        t.setStyleSheet(
            f"font-size:9px; color:{Colors.TEXT_MUTED}; letter-spacing:1px;"
            f" font-weight:600;"
        )
        layout.addWidget(t)

        v = QLabel(value)
        v.setStyleSheet(
            f"font-size:22px; font-weight:700;"
            f" color:{value_color or Colors.TEXT_PRIMARY};"
            f" letter-spacing:-0.5px;"
        )
        layout.addWidget(v)

        if subtitle:
            s = QLabel(subtitle)
            s.setStyleSheet(f"font-size:10px; color:{Colors.TEXT_MUTED};")
            layout.addWidget(s)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CriticalAlertWidget — Kritik bulgu uyarı kutusu
# ─────────────────────────────────────────────────────────────────────────────
class CriticalAlertWidget(QFrame):
    """
    Strok/kritik tespit uyarısı.
    Kullanım:
        alert = CriticalAlertWidget(
            title="Kritik Bulgu",
            message="Yüksek inme olasılığı (%79.4) tespit edildi."
        )
    """

    def __init__(self, title: str, message: str,
                 level: str = "danger", parent=None):
        super().__init__(parent)
        colors = {
            "danger":  (Colors.DANGER_BG,  Colors.DANGER_BORDER,  "#EF9A9A", Colors.DANGER),
            "warning": (Colors.WARNING_BG, "#4A3010",             "#FFCC80", Colors.WARNING),
            "success": (Colors.SUCCESS_BG, "#1B4A1D",             "#A5D6A7", Colors.SUCCESS),
        }
        bg, border, msg_color, title_color = colors.get(level, colors["danger"])

        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border:1px solid {border};"
            f" border-radius:6px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        t = QLabel(title)
        t.setStyleSheet(
            f"font-size:11px; font-weight:600; color:{title_color};"
        )
        layout.addWidget(t)

        m = QLabel(message)
        m.setWordWrap(True)
        m.setStyleSheet(f"font-size:11px; color:{msg_color};")
        layout.addWidget(m)
