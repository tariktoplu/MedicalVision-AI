# ui/theme.py
# Minimalist Dark Theme — Turan YZ / NeuroViva-AI
# Renk paleti, şirket logolarından türetilmiştir.

import os

# Logo dosya yolları
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_DARK  = os.path.join(_UI_DIR, "WhatsApp Image 2026-05-05 at 20.17.07.jpeg")
LOGO_LIGHT = os.path.join(_UI_DIR, "WhatsApp Image 2026-05-05 at 20.17.06.jpeg")


class Colors:
    # ── Ana arka planlar ──────────────────────────────────────────────────
    BG_PRIMARY   = "#0A0A0F"   # derin siyah
    BG_SIDEBAR   = "#0E0E14"   # sidebar / topbar
    BG_CARD      = "#141419"   # kart yüzeyleri
    BG_CARD2     = "#1A1A22"   # elevated kart / hover
    BG_SUBTLE    = "#20202A"   # input / ince vurgu

    # ── Kenarlıklar ───────────────────────────────────────────────────────
    BORDER       = "#1E1E28"
    BORDER_MED   = "#2A2A38"

    # ── Marka renkleri (Logolardan) ───────────────────────────────────────
    PRIMARY      = "#E63946"   # kırmızı — ana vurgu (koyu logo)
    PRIMARY_HOVER= "#FF4757"
    ACCENT       = "#7ECBC0"   # teal/mint — ikincil vurgu (açık logo)
    ACCENT_DIM   = "#5A9E95"   # teal karartılmış

    # ── Anlam renkleri ────────────────────────────────────────────────────
    DANGER       = "#E63946"
    DANGER_BG    = "#1A0A0C"
    DANGER_BORDER= "#3D1519"
    SUCCESS      = "#2ECC71"
    SUCCESS_BG   = "#0A1A0F"
    WARNING      = "#F39C12"
    WARNING_BG   = "#1A1505"

    # ── Metin ─────────────────────────────────────────────────────────────
    TEXT_PRIMARY  = "#F0F0F5"
    TEXT_SECONDARY= "#808096"
    TEXT_MUTED    = "#4A4A5A"

    # ── Modalite renkleri ─────────────────────────────────────────────────
    MR_COLOR     = "#7ECBC0"   # teal (logodaki mint)
    MR_BG        = "#0F1E1C"
    BT_COLOR     = "#E63946"   # kırmızı
    BT_BG        = "#1A0A0C"


DARK_THEME = f"""
/* ===== GENEL ===== */
QMainWindow, QWidget {{
    background-color: {Colors.BG_PRIMARY};
    color: {Colors.TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", "SF Pro Display", sans-serif;
    font-size: 14px;
}}

/* Label'lara hover etkisi olmasın */
QLabel {{
    background: transparent;
    border: none;
}}

/* ===== KAYDIRMA ÇUBUĞU ===== */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 4px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {Colors.BORDER_MED};
    border-radius: 2px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {Colors.TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 4px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {Colors.BORDER_MED};
    border-radius: 2px;
}}

/* ===== ÇERÇEVELER / KARTLAR ===== */
QFrame#card {{
    background-color: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER};
    border-radius: 8px;
}}
QFrame#card_elevated {{
    background-color: {Colors.BG_CARD2};
    border: 1px solid {Colors.BORDER_MED};
    border-radius: 8px;
}}

/* ===== BUTONLAR ===== */
QPushButton {{
    background-color: {Colors.BG_SUBTLE};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {Colors.BG_CARD2};
    border-color: {Colors.BORDER_MED};
}}
QPushButton:pressed {{
    background-color: {Colors.BORDER};
}}
QPushButton:disabled {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT_MUTED};
    border-color: {Colors.BORDER};
}}

QPushButton#btn_primary {{
    background-color: {Colors.PRIMARY};
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}}
QPushButton#btn_primary:hover {{
    background-color: {Colors.PRIMARY_HOVER};
}}
QPushButton#btn_primary:pressed {{
    background-color: #C62D3A;
}}

QPushButton#btn_danger {{
    background-color: {Colors.DANGER_BG};
    color: {Colors.DANGER};
    border: 1px solid {Colors.DANGER_BORDER};
}}
QPushButton#btn_danger:hover {{
    background-color: #200D0F;
}}

QPushButton#btn_success {{
    background-color: {Colors.SUCCESS_BG};
    color: {Colors.SUCCESS};
    border: 1px solid #1B4A1D;
}}
QPushButton#btn_success:hover {{
    background-color: #0F2A10;
}}

QPushButton#btn_back {{
    background-color: transparent;
    color: {Colors.TEXT_SECONDARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
}}
QPushButton#btn_back:hover {{
    color: {Colors.TEXT_PRIMARY};
    background-color: {Colors.BG_CARD};
}}

QPushButton#btn_accent {{
    background-color: {Colors.ACCENT};
    color: #0A0A0F;
    border: none;
    font-weight: 600;
}}
QPushButton#btn_accent:hover {{
    background-color: #8FD8CE;
}}

/* ===== ETİKETLER ===== */
QLabel#title_main {{
    font-size: 22px;
    font-weight: 700;
    color: {Colors.TEXT_PRIMARY};
    letter-spacing: -0.5px;
}}
QLabel#title_section {{
    font-size: 13px;
    font-weight: 600;
    color: {Colors.TEXT_PRIMARY};
}}
QLabel#label_muted {{
    font-size: 11px;
    color: {Colors.TEXT_SECONDARY};
}}
QLabel#breadcrumb {{
    font-size: 11px;
    color: {Colors.TEXT_MUTED};
}}
QLabel#badge_red {{
    background-color: {Colors.DANGER_BG};
    color: {Colors.DANGER};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel#badge_teal {{
    background-color: {Colors.MR_BG};
    color: {Colors.ACCENT};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel#badge_success {{
    background-color: {Colors.SUCCESS_BG};
    color: {Colors.SUCCESS};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel#badge_danger {{
    background-color: {Colors.DANGER_BG};
    color: #EF9A9A;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}

/* ===== TABLO ===== */
QTableWidget {{
    background-color: {Colors.BG_CARD};
    alternate-background-color: {Colors.BG_CARD2};
    gridline-color: {Colors.BORDER};
    border: 1px solid {Colors.BORDER};
    border-radius: 8px;
    color: {Colors.TEXT_PRIMARY};
    font-size: 13px;
    selection-background-color: {Colors.BG_CARD2};
}}
QTableWidget::item {{
    padding: 8px 12px;
}}
QTableWidget::item:selected {{
    background-color: {Colors.BG_SUBTLE};
    color: {Colors.TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {Colors.BORDER};
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ===== İLERLEME ÇUBUĞU ===== */
QProgressBar {{
    background-color: {Colors.BG_SUBTLE};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    font-size: 10px;
    color: {Colors.TEXT_SECONDARY};
}}
QProgressBar::chunk {{
    background-color: {Colors.PRIMARY};
    border-radius: 3px;
}}
QProgressBar#progress_success::chunk {{
    background-color: {Colors.SUCCESS};
}}

/* ===== AYIRICI ===== */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {Colors.BORDER};
}}
QSplitter::handle {{
    background-color: {Colors.BORDER};
    width: 1px;
}}

/* ===== GİRİŞ ALANLARI ===== */
QLineEdit, QTextEdit {{
    background-color: {Colors.BG_SUBTLE};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    color: {Colors.TEXT_PRIMARY};
    padding: 6px 10px;
    font-size: 12px;
    selection-background-color: {Colors.PRIMARY};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {Colors.ACCENT};
}}

/* ===== MESAJ KUTUSU ===== */
QMessageBox {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT_PRIMARY};
}}
QMessageBox QPushButton {{
    min-width: 80px;
}}

/* ===== DİYALOG ===== */
QDialog {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT_PRIMARY};
}}
QDialogButtonBox QPushButton {{
    min-width: 80px;
}}
QFormLayout QLabel {{
    color: {Colors.TEXT_SECONDARY};
}}

/* ===== STATUS BAR ===== */
QStatusBar {{
    background-color: {Colors.BG_PRIMARY};
    color: {Colors.TEXT_MUTED};
    border-top: 1px solid {Colors.BORDER};
    font-size: 12px;
    padding: 4px 16px;
}}
"""
