"""NihanCAD toolbar — tool selection, zoom, and action buttons."""

from PyQt6.QtWidgets import (
    QToolBar, QToolButton, QLabel, QLineEdit, QWidget, QHBoxLayout,
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QAction, QKeySequence


class CADToolBar(QToolBar):
    """Main toolbar with tool buttons, toggles, zoom control, and actions.

    Signals:
        tool_changed(str) — active tool name
        action_triggered(str) — one-shot action name
    """

    tool_changed = pyqtSignal(str)
    action_triggered = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__('Araçlar', parent)
        self.setObjectName('MainToolBar')
        self.setMovable(False)
        self.setIconSize(QSize(20, 20))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self._tool_buttons: dict[str, QToolButton] = {}
        self._toggle_buttons: dict[str, QToolButton] = {}

        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        # --- File action ---
        self._add_action_button('open', '\u2610 Ac', 'Dosya ac (Ctrl+O)')
        self.addSeparator()

        # --- Tools (exclusive group) ---
        self._add_tool_button('select', '\u271B Sec', 'Secim araci')
        self._add_tool_button('pan', '\u2B6E Pan', 'Gorunumu surukle (Space+Click)')
        self._add_tool_button('measure_dist', '\u2194 Olcu', 'Mesafe olc (M)')
        self._add_tool_button('measure_angle', '\u2220 Aci', 'Aci olc (A)')
        self.addSeparator()

        # --- Toggles ---
        self._add_toggle_button('grid', '\u2591 Grid', 'Izgara goster/gizle (G)')
        self._add_toggle_button('snap', '\u2316 Snap', 'Yakalama ac/kapat (S)')
        self.addSeparator()

        # --- View actions ---
        self._add_action_button('fit_all', '\u2B1A Sigdir', 'Tumu sigdir (F)')
        self._add_action_button('fit_selection', '\u2B95 Secime', 'Secili parcaya zum (Z)')

        # --- Zoom input ---
        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(6, 0, 6, 0)
        zoom_layout.setSpacing(3)

        zoom_label = QLabel('Zum:')
        zoom_label.setStyleSheet('color: #8b8fa8; font-size: 11px;')
        zoom_layout.addWidget(zoom_label)

        self._zoom_input = QLineEdit('100%')
        self._zoom_input.setFixedWidth(54)
        self._zoom_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_input.setToolTip('Zum orani (%)')
        self._zoom_input.returnPressed.connect(self._on_zoom_entered)
        zoom_layout.addWidget(self._zoom_input)

        self.addWidget(zoom_widget)
        self.addSeparator()

        # --- Export ---
        self._add_action_button('export_png', 'PNG', 'PNG olarak kaydet (Ctrl+E)')

        # Default tool
        self.set_active_tool('select')

    # ------------------------------------------------------------------
    # Button factories
    # ------------------------------------------------------------------

    def _make_button(self, text: str, tooltip: str, checkable: bool = False) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setCheckable(checkable)
        btn.setMinimumWidth(48)
        return btn

    def _add_tool_button(self, tool_id: str, text: str, tooltip: str):
        btn = self._make_button(text, tooltip, checkable=True)
        btn.clicked.connect(lambda checked, tid=tool_id: self._on_tool_clicked(tid))
        self.addWidget(btn)
        self._tool_buttons[tool_id] = btn

    def _add_toggle_button(self, action_id: str, text: str, tooltip: str):
        btn = self._make_button(text, tooltip, checkable=True)
        btn.clicked.connect(
            lambda checked, aid=action_id: self.action_triggered.emit(aid)
        )
        self.addWidget(btn)
        self._toggle_buttons[action_id] = btn

    def _add_action_button(self, action_id: str, text: str, tooltip: str):
        btn = self._make_button(text, tooltip)
        btn.clicked.connect(lambda: self.action_triggered.emit(action_id))
        self.addWidget(btn)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_tool_clicked(self, tool_id: str):
        self.set_active_tool(tool_id)
        self.tool_changed.emit(tool_id)

    def _on_zoom_entered(self):
        text = self._zoom_input.text().replace('%', '').strip()
        try:
            _ = float(text)
            self.action_triggered.emit(f'zoom:{text}')
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_tool(self, tool_id: str):
        """Set the active tool button (exclusive selection)."""
        for tid, btn in self._tool_buttons.items():
            btn.setChecked(tid == tool_id)

    def set_zoom_display(self, percent: float):
        """Update the zoom percentage display."""
        self._zoom_input.setText(f'{percent:.0f}%')

    def set_toggle_state(self, toggle_id: str, active: bool):
        """Set a toggle button's checked state."""
        btn = self._toggle_buttons.get(toggle_id)
        if btn:
            btn.setChecked(active)

    def is_toggled(self, toggle_id: str) -> bool:
        """Check if a toggle button is currently active."""
        btn = self._toggle_buttons.get(toggle_id)
        return btn.isChecked() if btn else False
