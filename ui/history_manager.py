# ui/history_manager.py
# Analiz geçmişi yönetimi — JSON dosyasına kayıt/okuma

import os
import json
from datetime import datetime

_HISTORY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(_HISTORY_DIR, "analysis_history.json")
MAX_HISTORY  = 50  # Son 50 analiz


def load_history() -> list:
    """Geçmiş analiz kayıtlarını yükle."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_entry(entry: dict):
    """Yeni bir analiz kaydı ekle."""
    history = load_history()
    entry["timestamp"] = datetime.now().strftime("%d.%m.%Y  %H:%M")
    history.insert(0, entry)  # en yenisi başta
    history = history[:MAX_HISTORY]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Geçmiş kaydedilemedi: {e}")


def clear_history():
    """Tüm geçmişi sil."""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception:
        pass
