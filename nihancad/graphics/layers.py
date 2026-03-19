"""Layer management for NihanCAD graphics engine.

Each visual element type (cutline, seamline, etc.) belongs to a layer
that controls visibility, color, opacity, and line style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class Layer:
    """Visual layer definition for a category of drawing elements.

    Attributes:
        id: Unique identifier ('cutline', 'seamline', etc.).
        label: Human-readable display name (Turkish).
        color: Default hex color string.
        visible: Whether items on this layer are drawn.
        opacity: Alpha [0.0, 1.0].
        line_width: Stroke width in pixels (cosmetic).
        dash: Dash pattern as list of floats, or None for solid.
    """

    id: str
    label: str
    color: str
    visible: bool = True
    opacity: float = 1.0
    line_width: float = 2.0
    dash: Optional[list[float]] = None


# ── Default layer set ──────────────────────────────────────────────

_DEFAULT_LAYERS: list[Layer] = [
    Layer("cutline", "Kesim Çizgisi", "#e94560", True, 0.8, 2.0),
    Layer("seamline", "Dikiş Çizgisi", "#00b4d8", True, 0.9, 1.0, [6, 4]),
    Layer("grainline", "Kumaş Yönü", "#48bb78", True, 1.0, 1.5),
    Layer("notch", "Çentik", "#ed8936", True, 1.0, 1.0),
    Layer("refline", "Referans", "#888888", True, 0.5, 0.5, [4, 4]),
    Layer("text", "Yazılar", "#aaaaaa", True, 1.0),
    Layer("internal", "İç Çizim", "#c792ea", True, 0.8, 1.0, [4, 3]),
    Layer("lining", "Astar", "#82aaff", True, 0.7, 1.5, [8, 4]),
    Layer("drill", "Delik Noktası", "#ffcb6b", True, 1.0, 1.0),
    Layer("annotation", "Üretim Notu", "#c3e88d", True, 0.8),
]


class LayerManager(QObject):
    """Manages the set of visual layers and emits change signals.

    Signals:
        layer_changed(str): Emitted with the layer id whenever a
            layer property is modified.
    """

    layer_changed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._layers: dict[str, Layer] = {}
        for layer in _DEFAULT_LAYERS:
            # Create independent copies so defaults are never mutated.
            self._layers[layer.id] = Layer(
                id=layer.id,
                label=layer.label,
                color=layer.color,
                visible=layer.visible,
                opacity=layer.opacity,
                line_width=layer.line_width,
                dash=list(layer.dash) if layer.dash else None,
            )

    # ── Queries ────────────────────────────────────────────────────

    def get_layer(self, layer_id: str) -> Layer | None:
        """Return the Layer with *layer_id*, or ``None``."""
        return self._layers.get(layer_id)

    def get_all(self) -> list[Layer]:
        """Return all layers in definition order."""
        return list(self._layers.values())

    # ── Mutations ──────────────────────────────────────────────────

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Toggle visibility of *layer_id*."""
        layer = self._layers.get(layer_id)
        if layer is None:
            return
        if layer.visible != visible:
            layer.visible = visible
            self.layer_changed.emit(layer_id)

    def set_color(self, layer_id: str, color: str) -> None:
        """Update the color of *layer_id* (hex string)."""
        layer = self._layers.get(layer_id)
        if layer is None:
            return
        if layer.color != color:
            layer.color = color
            self.layer_changed.emit(layer_id)

    def set_opacity(self, layer_id: str, opacity: float) -> None:
        """Update opacity of *layer_id* (clamped to [0, 1])."""
        layer = self._layers.get(layer_id)
        if layer is None:
            return
        opacity = max(0.0, min(1.0, opacity))
        if layer.opacity != opacity:
            layer.opacity = opacity
            self.layer_changed.emit(layer_id)

    def set_line_width(self, layer_id: str, width: float) -> None:
        """Update line width of *layer_id*."""
        layer = self._layers.get(layer_id)
        if layer is None:
            return
        if layer.line_width != width:
            layer.line_width = width
            self.layer_changed.emit(layer_id)
