"""NihanCAD layer panel — right sidebar for layer visibility and style."""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QSlider, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor


# Default layers when no LayerManager is connected
DEFAULT_LAYERS = [
    {'id': 'cutline',   'name': 'Kesim Cizgisi',  'color': '#e94560', 'visible': True,  'opacity': 1.0},
    {'id': 'seamline',  'name': 'Dikis Cizgisi',   'color': '#00b4d8', 'visible': True,  'opacity': 1.0},
    {'id': 'grainline', 'name': 'Kumas Yonu',      'color': '#48bb78', 'visible': True,  'opacity': 1.0},
    {'id': 'notch',     'name': 'Centik',           'color': '#ed8936', 'visible': True,  'opacity': 1.0},
    {'id': 'refline',   'name': 'Referans',         'color': '#888888', 'visible': True,  'opacity': 0.6},
    {'id': 'text',      'name': 'Yazilar',          'color': '#aaaaaa', 'visible': True,  'opacity': 0.8},
    {'id': 'internal',   'name': 'İç Çizim',       'color': '#c792ea', 'visible': True,  'opacity': 0.8},
    {'id': 'lining',     'name': 'Astar',           'color': '#82aaff', 'visible': True,  'opacity': 0.7},
    {'id': 'drill',      'name': 'Delik Noktası',   'color': '#ffcb6b', 'visible': True,  'opacity': 1.0},
    {'id': 'annotation', 'name': 'Üretim Notu',     'color': '#c3e88d', 'visible': True,  'opacity': 0.8},
]


class _LayerRow(QWidget):
    """Single layer row with visibility toggle, color indicator, name, and opacity slider."""

    visibility_changed = pyqtSignal(str, bool)
    opacity_changed = pyqtSignal(str, float)

    def __init__(self, layer_id: str, name: str, color: str,
                 visible: bool = True, opacity: float = 1.0, parent=None):
        super().__init__(parent)
        self.layer_id = layer_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Visibility checkbox
        self._vis_cb = QCheckBox()
        self._vis_cb.setChecked(visible)
        self._vis_cb.setToolTip('Gorunurluk')
        self._vis_cb.stateChanged.connect(self._on_vis_changed)
        layout.addWidget(self._vis_cb)

        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(12, 12)
        color_label.setStyleSheet(
            f'background-color: {color}; border-radius: 2px; border: 1px solid #3d3e5c;'
        )
        layout.addWidget(color_label)

        # Layer name
        name_label = QLabel(name)
        name_label.setStyleSheet('font-size: 11px; color: #e2e4f0;')
        layout.addWidget(name_label, 1)

        # Opacity slider
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(opacity * 100))
        self._opacity_slider.setFixedWidth(60)
        self._opacity_slider.setToolTip(f'Opaklık: {int(opacity * 100)}%')
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self._opacity_slider)

        # Opacity value label
        self._opacity_label = QLabel(f'{int(opacity * 100)}%')
        self._opacity_label.setStyleSheet('font-size: 10px; color: #8b8fa8; min-width: 28px;')
        self._opacity_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._opacity_label)

    def _on_vis_changed(self, state):
        self.visibility_changed.emit(self.layer_id, state == Qt.CheckState.Checked.value)

    def _on_opacity_changed(self, value):
        self._opacity_label.setText(f'{value}%')
        self._opacity_slider.setToolTip(f'Opaklık: {value}%')
        self.opacity_changed.emit(self.layer_id, value / 100.0)


class LayerPanel(QDockWidget):
    """Right sidebar — layer visibility and style control.

    Signals:
        visibility_changed(str, bool) — layer_id, visible
        opacity_changed(str, float) — layer_id, opacity 0-1
    """

    visibility_changed = pyqtSignal(str, bool)
    opacity_changed = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__('Katmanlar', parent)
        self.setObjectName('LayerPanel')
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Header
        header = QLabel('  KATMANLAR')
        header.setStyleSheet(
            'font-size: 10px; font-weight: 700; color: #8b8fa8; '
            'padding: 6px 8px; letter-spacing: 1px;'
        )
        self._layout.addWidget(header)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._layout.addWidget(self._rows_container)

        self._layout.addStretch()
        self.setWidget(self._container)

        # Load defaults
        self.load_layers(DEFAULT_LAYERS)

    def load_layers(self, layers: list):
        """Populate the panel from a list of layer dicts or Layer objects.

        Each item should have: id, name, color, visible (opt), opacity (opt).
        """
        # Clear existing rows
        while self._rows_layout.count():
            child = self._rows_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for layer in layers:
            if isinstance(layer, dict):
                lid = layer['id']
                name = layer['name']
                color = layer.get('color', '#888888')
                visible = layer.get('visible', True)
                opacity = layer.get('opacity', 1.0)
            else:
                # Assume Layer object attributes
                lid = layer.id
                name = layer.name
                color = getattr(layer, 'color', '#888888')
                visible = getattr(layer, 'visible', True)
                opacity = getattr(layer, 'opacity', 1.0)

            row = _LayerRow(lid, name, color, visible, opacity)
            row.visibility_changed.connect(self.visibility_changed)
            row.opacity_changed.connect(self.opacity_changed)
            self._rows_layout.addWidget(row)

            # Separator between layers
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet('color: #252640;')
            sep.setFixedHeight(1)
            self._rows_layout.addWidget(sep)
