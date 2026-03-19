"""NihanCAD properties panel — shows details of the selected piece."""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QGridLayout, QLabel, QFrame,
)
from PyQt6.QtCore import Qt


class _Separator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet('color: #3d3e5c;')
        self.setFixedHeight(1)


class PropertiesPanel(QDockWidget):
    """Bottom-left panel showing selected piece properties."""

    def __init__(self, parent=None):
        super().__init__('Ozellikler', parent)
        self.setObjectName('PropertiesPanel')
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._container = QWidget()
        self._main_layout = QVBoxLayout(self._container)
        self._main_layout.setContentsMargins(10, 8, 10, 8)
        self._main_layout.setSpacing(0)

        # Empty state
        self._empty_label = QLabel('Parca secin')
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            'color: #8b8fa8; font-size: 12px; padding: 20px; font-style: italic;'
        )

        # Properties grid
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)
        self._grid.setColumnStretch(1, 1)

        self._rows: list[tuple[QLabel, QLabel]] = []
        self._separators: list[_Separator] = []

        self._fields = [
            ('Parca Adi', 'name'),
            ('Beden', 'size'),
            ('Malzeme', 'material'),
            ('Adet', 'quantity'),
            None,  # separator
            ('Alan', 'area'),
            ('Cevre', 'perimeter'),
            ('Boyut', 'dimensions'),
            ('Vertex', 'vertices'),
            ('Kumas Yonu', 'grainline'),
            ('Centik', 'notches'),
        ]

        row = 0
        for field in self._fields:
            if field is None:
                sep = _Separator()
                self._grid.addWidget(sep, row, 0, 1, 2)
                self._separators.append(sep)
                row += 1
                continue

            label_text, key = field
            key_label = QLabel(label_text)
            key_label.setStyleSheet('color: #8b8fa8; font-size: 11px;')
            key_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )

            val_label = QLabel('-')
            val_label.setStyleSheet('color: #e2e4f0; font-size: 11px; font-weight: 500;')
            val_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
            )
            val_label.setWordWrap(True)

            self._grid.addWidget(key_label, row, 0)
            self._grid.addWidget(val_label, row, 1)
            self._rows.append((key_label, val_label))
            row += 1

        self._main_layout.addWidget(self._empty_label)
        self._main_layout.addWidget(self._grid_widget)
        self._main_layout.addStretch()
        self._grid_widget.hide()

        self.setWidget(self._container)

    def update_piece(self, piece):
        """Display properties of the given Piece object."""
        if piece is None:
            self.clear()
            return

        self._empty_label.hide()
        self._grid_widget.show()

        # Build values list matching _fields (skip None separators)
        values = []
        values.append(piece.name or f'Parca #{piece.id}')
        values.append(piece.size or '-')
        values.append(piece.material or '-')
        values.append(str(piece.quantity))
        # separator skipped
        values.append(f'{piece.area:.1f} cm\u00b2' if piece.area else '-')
        values.append(f'{piece.perimeter:.1f} mm' if piece.perimeter else '-')

        w = piece.width if piece.width else 0
        h = piece.height if piece.height else 0
        values.append(f'{w:.1f} x {h:.1f} mm')

        vertex_count = len(piece.cutline) if piece.cutline else 0
        values.append(str(vertex_count))

        # Grainline angle (Grainline dataclass has .angle in degrees)
        if piece.grainlines and len(piece.grainlines) > 0:
            gl = piece.grainlines[0]
            angle = gl.angle % 360
            values.append(f'{angle:.0f}\u00b0')
        else:
            values.append('-')

        notch_count = len(piece.notches) if piece.notches else 0
        values.append(f'{notch_count} adet')

        for i, (_, val_label) in enumerate(self._rows):
            val_label.setText(values[i] if i < len(values) else '-')

    def clear(self):
        """Show the empty state."""
        self._grid_widget.hide()
        self._empty_label.show()
