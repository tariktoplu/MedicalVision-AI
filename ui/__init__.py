# ui/__init__.py
from .theme import DARK_THEME, Colors, LOGO_DARK, LOGO_LIGHT
from .custom_widgets import (
    CircularProgressWidget,
    ProbabilityBarWidget,
    SkeletonWidget,
    DragDropZone,
    ResultCard,
    CriticalAlertWidget,
)
from .history_manager import load_history, save_entry, clear_history
