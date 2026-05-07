# pages/start_page.py
# Ana Sayfa — Logo + Modalite Seçimi + Geçmiş Analizler

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

from ui.theme import Colors, LOGO_DARK
from ui.history_manager import load_history, clear_history


class ModalityCard(QFrame):
    """BT veya MR seçim kartı."""
    clicked = pyqtSignal(str)

    def __init__(self, modality: str, title: str, subtitle: str,
                 accent: str, accent_bg: str, class_count: str,
                 parent=None):
        super().__init__(parent)
        self._modality = modality
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setStyleSheet(
            f"QFrame#modalityCard {{ background:{Colors.BG_CARD}; border:1px solid {Colors.BORDER};"
            f" border-radius:10px; }}"
            f"QFrame#modalityCard:hover {{ border-color:{accent}; background:{Colors.BG_CARD2}; }}"
        )
        self.setObjectName("modalityCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        # Üst satır: tag + başlık + ok
        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        tag = QLabel(modality)
        tag.setFixedSize(42, 42)
        tag.setAlignment(Qt.AlignCenter)
        tag.setStyleSheet(
            f"background:{accent_bg}; color:{accent}; border-radius:8px;"
            f" font-size:16px; font-weight:700; border:none;"
        )
        top_row.addWidget(tag)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size:18px; font-weight:600; color:{Colors.TEXT_PRIMARY};"
            " border:none; background:transparent;"
        )
        top_row.addWidget(title_lbl)
        top_row.addStretch()

        arrow = QLabel("→")
        arrow.setStyleSheet(
            f"font-size:20px; color:{Colors.TEXT_MUTED}; border:none; background:transparent;"
        )
        top_row.addWidget(arrow)
        layout.addLayout(top_row)

        # Açıklama
        desc_lbl = QLabel(subtitle)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(
            f"font-size:13px; color:{Colors.TEXT_SECONDARY}; border:none; background:transparent;"
        )
        layout.addWidget(desc_lbl)

        # Alt badge
        badge = QLabel(class_count)
        badge.setStyleSheet(
            f"font-size:11px; color:{accent}; font-weight:600;"
            f" letter-spacing:0.5px; border:none; background:transparent;"
        )
        layout.addWidget(badge)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._modality)
        super().mousePressEvent(event)


class HistoryRow(QFrame):
    """Geçmiş analiz satırı."""
    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame#histRow {{ background:transparent; border:none;"
            f" border-bottom:1px solid {Colors.BORDER}; }}"
        )
        self.setObjectName("histRow")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        # Dosya adı
        fname = QLabel(entry.get("filename", "—"))
        fname.setStyleSheet(
            f"font-size:13px; color:{Colors.TEXT_PRIMARY}; border:none;"
        )
        layout.addWidget(fname, 3)

        # Modalite badge
        modality = entry.get("modality", "?")
        mod_color = Colors.PRIMARY if modality == "BT" else Colors.ACCENT
        mod_bg = Colors.BT_BG if modality == "BT" else Colors.MR_BG
        mod_lbl = QLabel(f"  {modality}  ")
        mod_lbl.setFixedWidth(44)
        mod_lbl.setAlignment(Qt.AlignCenter)
        mod_lbl.setStyleSheet(
            f"background:{mod_bg}; color:{mod_color}; border-radius:4px;"
            f" font-size:11px; font-weight:600; border:none;"
        )
        layout.addWidget(mod_lbl)

        # Sonuç
        result = entry.get("result", "—")
        is_critical = any(c in result for c in ("İnme", "HiperakutAkut", "Subakut"))
        res_color = Colors.DANGER if is_critical else Colors.SUCCESS
        res_bg = Colors.DANGER_BG if is_critical else Colors.SUCCESS_BG
        res_lbl = QLabel(f"  {result}  ")
        res_lbl.setAlignment(Qt.AlignCenter)
        res_lbl.setStyleSheet(
            f"background:{res_bg}; color:{res_color}; border-radius:4px;"
            f" font-size:11px; font-weight:600; border:none;"
        )
        layout.addWidget(res_lbl, 2)

        # Olasılık
        prob = entry.get("confidence", "")
        if prob:
            prob_lbl = QLabel(f"%{prob}")
            prob_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            prob_lbl.setStyleSheet(
                f"font-size:12px; font-weight:600; color:{res_color}; border:none;"
            )
            layout.addWidget(prob_lbl, 1)

        # Tarih
        date = entry.get("timestamp", "")
        date_lbl = QLabel(date)
        date_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        date_lbl.setStyleSheet(
            f"font-size:11px; color:{Colors.TEXT_MUTED}; border:none;"
        )
        layout.addWidget(date_lbl, 2)


# ─────────────────────────────────────────────────────────────────────────────
class StartPage(QWidget):
    """Ana sayfa — Logo + Modalite Seçimi + Geçmiş Analizler."""
    modality_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Scroll ───────────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")

        content = QWidget()
        content.setStyleSheet(f"background:{Colors.BG_PRIMARY};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(48, 36, 48, 36)
        content_layout.setSpacing(0)

        # ── Üst Bölüm: Logo + Başlık ────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(20)

        # Logo
        logo_label = QLabel()
        logo_label.setFixedSize(80, 80)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("background:transparent; border:none;")
        if os.path.exists(LOGO_DARK):
            pixmap = QPixmap(LOGO_DARK)
            scaled = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled)
        header.addWidget(logo_label)

        # Başlık grubu
        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        company = QLabel("TURAN YZ")
        company.setStyleSheet(
            f"font-size:11px; font-weight:700; color:{Colors.PRIMARY};"
            f" letter-spacing:3px; border:none;"
        )
        title_col.addWidget(company)

        title = QLabel("NeuroViva AI")
        title.setStyleSheet(
            f"font-size:32px; font-weight:700; color:{Colors.TEXT_PRIMARY};"
            f" letter-spacing:-0.5px; border:none;"
        )
        title_col.addWidget(title)

        subtitle = QLabel("Beyin Görüntü Analiz Sistemi  •  ConvNeXt-Tiny Ensemble")
        subtitle.setStyleSheet(
            f"font-size:14px; color:{Colors.TEXT_MUTED}; border:none;"
        )
        title_col.addWidget(subtitle)

        header.addLayout(title_col, 1)
        content_layout.addLayout(header)
        content_layout.addSpacing(32)

        # ── Ayırıcı ─────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{Colors.BORDER}; border:none;")
        content_layout.addWidget(sep)
        content_layout.addSpacing(28)

        # ── Modalite Bölümü ──────────────────────────────────────────────────
        section_label = QLabel("GÖRÜNTÜ MODALİTESİ")
        section_label.setStyleSheet(
            f"font-size:11px; font-weight:700; color:{Colors.TEXT_MUTED};"
            f" letter-spacing:2px; border:none;"
        )
        content_layout.addWidget(section_label)
        content_layout.addSpacing(14)

        # Kartlar yan yana
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        bt_card = ModalityCard(
            modality="BT",
            title="Bilgisayarlı Tomografi",
            subtitle="DICOM beyin BT görüntülerini analiz eder.\nİnme / Sağlıklı sınıflandırması.",
            accent=Colors.PRIMARY,
            accent_bg=Colors.BT_BG,
            class_count="2 SINIF  •  İNME TESPİTİ",
        )
        bt_card.clicked.connect(self.modality_selected)

        mr_card = ModalityCard(
            modality="MR",
            title="Manyetik Rezonans",
            subtitle="MR görüntülerinde inme evrelerini sınıflandırır.\nHiperakutAkut, Subakut, NormalKronik.",
            accent=Colors.ACCENT,
            accent_bg=Colors.MR_BG,
            class_count="3 SINIF  •  EVRE SINIFLANDIRMASI",
        )
        mr_card.clicked.connect(self.modality_selected)

        cards_row.addWidget(bt_card)
        cards_row.addWidget(mr_card)
        content_layout.addLayout(cards_row)
        content_layout.addSpacing(32)

        # ── Geçmiş Analizler Bölümü ──────────────────────────────────────────
        history_header = QHBoxLayout()
        history_title = QLabel("GEÇMİŞ ANALİZLER")
        history_title.setStyleSheet(
            f"font-size:11px; font-weight:700; color:{Colors.TEXT_MUTED};"
            f" letter-spacing:2px; border:none;"
        )
        history_header.addWidget(history_title)
        history_header.addStretch()

        self._clear_btn = QPushButton("Temizle")
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setFixedWidth(80)
        self._clear_btn.setCursor(Qt.PointingHandCursor)
        self._clear_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{Colors.TEXT_MUTED};"
            f" border:1px solid {Colors.BORDER}; border-radius:4px; font-size:11px; }}"
            f"QPushButton:hover {{ color:{Colors.DANGER}; border-color:{Colors.DANGER_BORDER}; }}"
        )
        self._clear_btn.clicked.connect(self._on_clear_history)
        history_header.addWidget(self._clear_btn)

        content_layout.addLayout(history_header)
        content_layout.addSpacing(10)

        # Geçmiş içerik alanı
        self._history_container = QFrame()
        self._history_container.setObjectName("card")
        self._history_container.setStyleSheet(
            f"QFrame#card {{ background:{Colors.BG_CARD}; border:1px solid {Colors.BORDER};"
            f" border-radius:10px; }}"
        )
        self._history_layout = QVBoxLayout(self._history_container)
        self._history_layout.setContentsMargins(0, 0, 0, 0)
        self._history_layout.setSpacing(0)

        content_layout.addWidget(self._history_container)
        content_layout.addSpacing(24)

        # Footer
        footer = QLabel("Turan YZ © 2026")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f"font-size:10px; color:{Colors.TEXT_MUTED}; border:none;"
        )
        content_layout.addWidget(footer)
        content_layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # İlk yükleme
        self._refresh_history()

    def _refresh_history(self):
        """Geçmiş listeyi güncelle."""
        # Mevcut içeriği temizle
        while self._history_layout.count():
            item = self._history_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        history = load_history()

        if not history:
            empty = QLabel("Henüz analiz yapılmadı")
            empty.setAlignment(Qt.AlignCenter)
            empty.setFixedHeight(80)
            empty.setStyleSheet(
                f"font-size:13px; color:{Colors.TEXT_MUTED}; border:none;"
            )
            self._history_layout.addWidget(empty)
            self._clear_btn.setVisible(False)
            return

        self._clear_btn.setVisible(True)

        # Tablo başlığı
        header_frame = QFrame()
        header_frame.setFixedHeight(36)
        header_frame.setStyleSheet(
            f"background:{Colors.BG_CARD2}; border:none;"
            f" border-top-left-radius:10px; border-top-right-radius:10px;"
        )
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(16, 0, 16, 0)
        h_layout.setSpacing(12)

        for text, stretch in [("Dosya", 3), ("Tür", 0), ("Sonuç", 2), ("Güven", 1), ("Tarih", 2)]:
            lbl = QLabel(text.upper())
            lbl.setStyleSheet(
                f"font-size:10px; font-weight:600; color:{Colors.TEXT_MUTED};"
                f" letter-spacing:0.5px; border:none;"
            )
            if stretch:
                h_layout.addWidget(lbl, stretch)
            else:
                lbl.setFixedWidth(44)
                lbl.setAlignment(Qt.AlignCenter)
                h_layout.addWidget(lbl)

        self._history_layout.addWidget(header_frame)

        # Satırlar (son 15)
        for entry in history[:15]:
            row = HistoryRow(entry)
            self._history_layout.addWidget(row)

    def _on_clear_history(self):
        clear_history()
        self._refresh_history()

    def showEvent(self, event):
        """Sayfa her gösterildiğinde geçmişi yenile."""
        super().showEvent(event)
        self._refresh_history()
