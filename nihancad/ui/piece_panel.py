"""NihanCAD piece list panel — left sidebar showing all pieces."""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QHBoxLayout, QAbstractItemView,
)
from PyQt6.QtGui import QColor, QPixmap, QPainter, QIcon, QBrush, QPen
from PyQt6.QtCore import pyqtSignal, Qt, QSize


class PieceListItem(QWidget):
    """Custom widget for a single piece row in the list."""

    def __init__(self, piece, parent=None):
        super().__init__(parent)
        self.piece_id = piece.id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # Color swatch
        swatch = QLabel()
        pm = QPixmap(14, 14)
        pm.fill(QColor(piece.color if piece.color else '#e94560'))
        swatch.setPixmap(pm)
        swatch.setFixedSize(14, 14)
        layout.addWidget(swatch)

        # Info column
        info = QVBoxLayout()
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(1)

        # Name + size
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_label = QLabel(piece.name or f'Parca #{piece.id}')
        name_label.setStyleSheet('font-weight: 600; font-size: 12px; color: #e2e4f0;')
        name_row.addWidget(name_label)

        if piece.size:
            size_label = QLabel(piece.size)
            size_label.setStyleSheet(
                'font-size: 10px; color: #ffffff; background: #e94560; '
                'border-radius: 3px; padding: 0 4px; font-weight: 600;'
            )
            size_label.setFixedHeight(16)
            name_row.addWidget(size_label)

        name_row.addStretch()
        info.addLayout(name_row)

        # Dimensions
        w = piece.width if piece.width else 0
        h = piece.height if piece.height else 0
        dim_label = QLabel(f'{w:.1f} x {h:.1f} mm')
        dim_label.setStyleSheet('font-size: 10px; color: #8b8fa8;')
        info.addWidget(dim_label)

        layout.addLayout(info, 1)


class PiecePanel(QDockWidget):
    """Left sidebar listing all loaded pieces.

    Signals:
        piece_clicked(int) — piece id on single click
        piece_double_clicked(int) — piece id on double click (zoom-to)
    """

    piece_clicked = pyqtSignal(int)
    piece_double_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__('Parcalar', parent)
        self.setObjectName('PiecePanel')
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._pieces: dict[int, object] = {}
        self._item_map: dict[int, QListWidgetItem] = {}

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel('  PARCALAR')
        header.setStyleSheet(
            'font-size: 10px; font-weight: 700; color: #8b8fa8; '
            'padding: 6px 8px; letter-spacing: 1px;'
        )
        layout.addWidget(header)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        # Count label
        self._count_label = QLabel('0 parca')
        self._count_label.setStyleSheet(
            'font-size: 10px; color: #8b8fa8; padding: 4px 8px; '
            'border-top: 1px solid #3d3e5c;'
        )
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._count_label)

        self.setWidget(container)

    def load_pieces(self, pieces: list):
        """Populate the list from a list of Piece objects."""
        self._list.clear()
        self._pieces.clear()
        self._item_map.clear()

        for piece in pieces:
            self._pieces[piece.id] = piece
            item = QListWidgetItem()
            widget = PieceListItem(piece)
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, piece.id)
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
            self._item_map[piece.id] = item

        self._count_label.setText(f'{len(pieces)} parca')

    def set_selected(self, piece_id: int):
        """Highlight a piece by id. Pass -1 to deselect."""
        if piece_id < 0:
            self._list.clearSelection()
            self._list.setCurrentItem(None)
            return
        item = self._item_map.get(piece_id)
        if item:
            self._list.blockSignals(True)
            self._list.setCurrentItem(item)
            self._list.blockSignals(False)

    def _on_current_changed(self, current, previous):
        if current:
            pid = current.data(Qt.ItemDataRole.UserRole)
            if pid is not None:
                self.piece_clicked.emit(pid)

    def _on_double_click(self, item):
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid is not None:
            self.piece_double_clicked.emit(pid)
