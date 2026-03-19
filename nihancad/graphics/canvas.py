"""CAD Canvas — AutoCAD-like infinite canvas with smooth pan/zoom.

Built on ``QGraphicsView`` with a flipped Y-axis (``scale(1, -1)``)
so that DXF coordinates map directly to scene coordinates.

Interaction:
    * **Pan**: right-click drag / middle-click drag / Space + left-drag.
    * **Zoom**: mouse wheel centred on cursor.
    * **Select**: left-click on a piece.
    * **Double-click**: zoom-to-piece.

Grid auto-scales through nice intervals (0.1 … 1000 mm) based on
the current zoom level.  A scale-bar overlay in the bottom-right
corner shows the real mm measurement.
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from nihancad.graphics.items import (
    CutlineItem, GrainlineItem, MeasureOverlay, NotchItem,
    PieceGroup, PieceTextItem, RefLineItem, SeamlineItem, SnapIndicator,
)
from nihancad.graphics.layers import LayerManager

# Nice grid intervals in mm.
_NICE_INTERVALS: list[float] = [
    0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
]


def _pick_grid_interval(pixels_per_mm: float, min_gap_px: int = 12) -> float:
    """Choose the smallest nice interval whose screen gap >= *min_gap_px*."""
    for iv in _NICE_INTERVALS:
        if iv * pixels_per_mm >= min_gap_px:
            return iv
    return _NICE_INTERVALS[-1]


class CADCanvas(QGraphicsView):
    """AutoCAD-like infinite canvas with smooth pan/zoom.

    Signals:
        mouse_moved(float, float) — world (DXF) coordinates for status bar.
        piece_selected(int) — piece id of the clicked piece.
        piece_hovered(int) — piece id currently under cursor (-1 if none).
        zoom_changed(float) — current zoom factor after change.
    """

    mouse_moved = pyqtSignal(float, float)
    piece_selected = pyqtSignal(int)
    piece_hovered = pyqtSignal(int)
    zoom_changed = pyqtSignal(float)

    # ── Zoom limits ────────────────────────────────────────────────

    ZOOM_MIN: float = 0.005
    ZOOM_MAX: float = 200.0
    ZOOM_FACTOR: float = 1.15  # per wheel step

    def __init__(self, parent: QWidget | None = None) -> None:
        scene = QGraphicsScene(parent)
        # Generous default scene rect — auto-expands as items are added.
        scene.setSceneRect(-50_000, -50_000, 100_000, 100_000)
        super().__init__(scene, parent)

        # ── Rendering settings ─────────────────────────────────────
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Flip Y-axis so DXF coordinates (Y-up) render correctly.
        self.scale(1, -1)

        # ── Internal state ─────────────────────────────────────────
        self._tool: str = "select"  # 'select', 'pan', 'measure_dist', 'measure_angle'
        self._panning: bool = False
        self._pan_start: QPointF = QPointF()
        self._space_held: bool = False
        self._zoom_level: float = 1.0

        # Measure state
        self._measure_points: list[QPointF] = []
        self._measure_overlay: MeasureOverlay = MeasureOverlay()
        scene.addItem(self._measure_overlay)

        self._snap_indicator: SnapIndicator = SnapIndicator()
        scene.addItem(self._snap_indicator)

        # ── Appearance ─────────────────────────────────────────────
        self.setBackgroundBrush(QBrush(QColor("#1a1a2e")))
        self.setMouseTracking(True)

        # ── Layer manager & grid state ────────────────────────────
        self.layer_manager = LayerManager()
        self.layer_manager.layer_changed.connect(self._on_layer_changed)
        self._grid_visible: bool = True
        self.snap_enabled: bool = False
        self._piece_groups: list[PieceGroup] = []
        self._text_items: list[PieceTextItem] = []

    # ── Public API ─────────────────────────────────────────────────

    def load_pieces(self, pieces) -> None:
        """Load Piece objects into the scene as QGraphicsItems."""
        # Clear old items (keep measure overlay and snap indicator)
        for pg in self._piece_groups:
            self.scene().removeItem(pg)
        # Clear standalone text items from previous load
        for item in self._text_items:
            self.scene().removeItem(item)
        self._piece_groups.clear()
        self._text_items.clear()

        # Resolve layers once
        cutline_layer = self.layer_manager.get_layer('cutline')
        seamline_layer = self.layer_manager.get_layer('seamline')
        grainline_layer = self.layer_manager.get_layer('grainline')
        notch_layer = self.layer_manager.get_layer('notch')
        refline_layer = self.layer_manager.get_layer('refline')
        text_layer = self.layer_manager.get_layer('text')

        for piece in pieces:
            group = PieceGroup(piece.id, piece_data=piece)

            # Cutline
            if piece.cutline:
                item = CutlineItem(piece.cutline, piece.color, layer=cutline_layer)
                group.addToGroup(item)

            # Seamline
            if piece.seamline and len(piece.seamline) >= 2:
                item = SeamlineItem(piece.seamline, layer=seamline_layer)
                group.addToGroup(item)

            # Grainlines (GrainlineItem takes list of Grainline dataclasses)
            if piece.grainlines:
                item = GrainlineItem(piece.grainlines, layer=grainline_layer)
                group.addToGroup(item)

            # Notches (NotchItem takes list of Notch dataclasses)
            if piece.notches:
                item = NotchItem(piece.notches, layer=notch_layer)
                group.addToGroup(item)

            # Reference lines
            for rl in piece.ref_lines:
                item = RefLineItem(rl.x1, rl.y1, rl.x2, rl.y2, layer=refline_layer)
                group.addToGroup(item)

            self.scene().addItem(group)
            self._piece_groups.append(group)

            # Piece text — added as top-level scene item (NOT in group)
            # to avoid ItemIgnoresTransformations breaking group boundingRect.
            if piece.name:
                cx, cy = piece.centroid
                color = text_layer.color if text_layer else piece.color
                text_item = PieceTextItem(piece.name, cx, cy, color)
                text_item.setZValue(100)  # above cutline fill
                self.scene().addItem(text_item)
                self._text_items.append(text_item)

    def fit_all(self, margin: float = 40.0) -> None:
        """Alias for fitAll (snake_case)."""
        self.fitAll(margin)

    def fit_to_piece(self, piece_id: int, margin: float = 30.0) -> None:
        """Alias for fitToPiece (snake_case)."""
        self.fitToPiece(piece_id, margin)

    def set_tool(self, tool: str) -> None:
        """Alias for setTool (snake_case)."""
        self.setTool(tool)

    def select_piece(self, piece_id: int) -> None:
        """Programmatically select a piece."""
        self.piece_selected.emit(piece_id)

    def set_grid_visible(self, visible: bool) -> None:
        """Toggle grid visibility."""
        self._grid_visible = visible
        self.viewport().update()

    def toggle_grid(self) -> None:
        self._grid_visible = not self._grid_visible
        self.viewport().update()

    def toggle_snap(self) -> None:
        self.snap_enabled = not self.snap_enabled

    def get_zoom_percent(self) -> float:
        return self._zoom_level * 100.0

    def set_zoom_percent(self, pct: float) -> None:
        if pct <= 0:
            return
        factor = (pct / 100.0) / self._zoom_level
        center = QPointF(self.viewport().width() / 2, self.viewport().height() / 2)
        self._apply_zoom(factor, center)

    def export_png(self, path: str) -> None:
        """Export current view to PNG."""
        from PyQt6.QtGui import QImage
        vp = self.viewport()
        img = QImage(vp.size(), QImage.Format.Format_ARGB32)
        img.fill(QColor("#1a1a2e"))
        p = QPainter(img)
        self.render(p)
        p.end()
        img.save(path)

    def refresh_layers(self) -> None:
        """Refresh item visibility/style based on layer manager state."""
        for pg in self._piece_groups:
            for child in pg.childItems():
                layer_id = None
                if isinstance(child, CutlineItem):
                    layer_id = 'cutline'
                elif isinstance(child, SeamlineItem):
                    layer_id = 'seamline'
                elif isinstance(child, GrainlineItem):
                    layer_id = 'grainline'
                elif isinstance(child, NotchItem):
                    layer_id = 'notch'
                elif isinstance(child, RefLineItem):
                    layer_id = 'refline'
                if layer_id:
                    layer = self.layer_manager.get_layer(layer_id)
                    if layer:
                        child.setVisible(layer.visible)
                        child.update_layer(layer)

        # Standalone text items (not in groups)
        text_layer = self.layer_manager.get_layer('text')
        if text_layer:
            for ti in self._text_items:
                ti.setVisible(text_layer.visible)
                ti.update_layer(text_layer)

        self.viewport().update()

    def _on_layer_changed(self, layer_id: str) -> None:
        self.refresh_layers()

    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    def setTool(self, tool_name: str) -> None:  # noqa: N802
        """Switch active tool: 'select', 'pan', 'measure_dist', 'measure_angle'."""
        self._tool = tool_name
        self._measure_points.clear()
        self._measure_overlay.clear()
        self._snap_indicator.hide()

        if tool_name == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif tool_name in ("measure_dist", "measure_angle"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def fitAll(self, margin: float = 40.0) -> None:  # noqa: N802
        """Zoom to fit all scene items with a pixel margin."""
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        rect.adjust(-margin, -margin, margin, margin)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._sync_zoom_level()

    def fitToPiece(self, piece_id: int, margin: float = 30.0) -> None:  # noqa: N802
        """Zoom to the bounding rect of a specific piece."""
        for item in self.scene().items():
            if isinstance(item, PieceGroup) and item.piece_id == piece_id:
                rect = item.boundingRect()
                rect = item.mapRectToScene(rect)
                rect.adjust(-margin, -margin, margin, margin)
                self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
                self._sync_zoom_level()
                return

    @property
    def snap_indicator(self) -> SnapIndicator:
        return self._snap_indicator

    @property
    def measure_overlay(self) -> MeasureOverlay:
        return self._measure_overlay

    # ── Zoom helpers ───────────────────────────────────────────────

    def _apply_zoom(self, factor: float, center: QPointF) -> None:
        """Apply zoom *factor* centred on *center* (viewport coords)."""
        new_level = self._zoom_level * factor
        if new_level < self.ZOOM_MIN or new_level > self.ZOOM_MAX:
            return

        # Map the centre point to scene coords before and after scaling.
        old_scene = self.mapToScene(center.toPoint())
        self.scale(factor, factor)
        new_scene = self.mapToScene(center.toPoint())
        delta = new_scene - old_scene
        self.translate(delta.x(), delta.y())

        self._zoom_level = new_level
        self.zoom_changed.emit(self._zoom_level)

    def _sync_zoom_level(self) -> None:
        """Sync internal zoom level from current transform matrix."""
        t = self.transform()
        self._zoom_level = math.hypot(t.m11(), t.m12())
        self.zoom_changed.emit(self._zoom_level)

    def _pixels_per_mm(self) -> float:
        """How many viewport pixels correspond to 1 mm in scene."""
        p0 = self.mapFromScene(QPointF(0, 0))
        p1 = self.mapFromScene(QPointF(1, 0))
        return math.hypot(p1.x() - p0.x(), p1.y() - p0.y())

    # ── Grid drawing ───────────────────────────────────────────────

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        """Draw auto-scaling grid lines."""
        super().drawBackground(painter, rect)

        if not self._grid_visible:
            return

        ppm = self._pixels_per_mm()
        if ppm < 0.5:
            return  # too zoomed out — skip grid entirely

        minor = _pick_grid_interval(ppm, min_gap_px=12)
        major = minor * 10 if minor * 10 * ppm < 4000 else minor * 5

        left = math.floor(rect.left() / minor) * minor
        top = math.floor(rect.top() / minor) * minor
        right = rect.right()
        bottom = rect.bottom()

        # Minor grid
        minor_pen = QPen(QColor(255, 255, 255, 18), 0)
        painter.setPen(minor_pen)
        x = left
        while x <= right:
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += minor
        y = top
        while y <= bottom:
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += minor

        # Major grid
        major_pen = QPen(QColor(255, 255, 255, 40), 0)
        painter.setPen(major_pen)
        x = math.floor(rect.left() / major) * major
        while x <= right:
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += major
        y = math.floor(rect.top() / major) * major
        while y <= bottom:
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += major

    # ── Scale bar overlay ──────────────────────────────────────────

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        """Draw a scale-bar overlay in the bottom-right corner."""
        super().drawForeground(painter, rect)

        ppm = self._pixels_per_mm()
        if ppm < 0.01:
            return

        # Pick a bar that is 100–200 px wide.
        target_px = 150
        raw_mm = target_px / ppm
        # Round to a nice value.
        nice_mm = 1.0
        for iv in _NICE_INTERVALS:
            if iv >= raw_mm * 0.5:
                nice_mm = iv
                break
        bar_px = nice_mm * ppm

        # Viewport coordinates — we need to reset the painter transform.
        painter.save()
        painter.resetTransform()

        vw = self.viewport().width()
        vh = self.viewport().height()

        margin = 16
        bar_h = 8
        x0 = vw - margin - bar_px
        y0 = vh - margin - bar_h

        # Background
        bg_rect = QRectF(x0 - 10, y0 - 20, bar_px + 20, bar_h + 30)
        painter.setPen(QPen(QColor(200, 200, 200, 180), 1))
        painter.setBrush(QBrush(QColor(30, 30, 50, 200)))
        painter.drawRoundedRect(bg_rect, 4, 4)

        # Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(220, 220, 220)))
        painter.drawRect(QRectF(x0, y0, bar_px, bar_h))

        # Ticks at ends
        tick_pen = QPen(QColor(220, 220, 220), 1)
        painter.setPen(tick_pen)
        painter.drawLine(QPointF(x0, y0 - 3), QPointF(x0, y0 + bar_h + 3))
        painter.drawLine(QPointF(x0 + bar_px, y0 - 3), QPointF(x0 + bar_px, y0 + bar_h + 3))

        # Label
        label = f"{nice_mm:g} mm"
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 220))
        text_rect = QRectF(x0, y0 - 18, bar_px, 16)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.restore()

    # ── Mouse events ───────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Zoom centred on cursor position."""
        degrees = event.angleDelta().y() / 8
        steps = degrees / 15
        factor = self.ZOOM_FACTOR ** steps
        center = QPointF(event.position())
        self._apply_zoom(factor, center)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        btn = event.button()

        # Pan triggers: right-click, middle-click, or Space+left-click.
        if btn in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or (
            btn == Qt.MouseButton.LeftButton and self._space_held
        ) or self._tool == "pan":
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if btn == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            # Measurement tools
            if self._tool in ("measure_dist", "measure_angle"):
                self._handle_measure_click(scene_pos)
                event.accept()
                return

            # Selection
            item = self.scene().itemAt(scene_pos, self.transform())
            group = self._find_piece_group(item)
            if group is not None:
                self.piece_selected.emit(group.piece_id)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.translate(delta.x(), delta.y())
            event.accept()
            return

        # Track world coordinates.
        scene_pos = self.mapToScene(event.pos())
        self.mouse_moved.emit(scene_pos.x(), scene_pos.y())

        # Hover detection.
        item = self.scene().itemAt(scene_pos, self.transform())
        group = self._find_piece_group(item)
        if group is not None:
            self.piece_hovered.emit(group.piece_id)
        else:
            self.piece_hovered.emit(-1)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            if self._tool == "pan":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self._tool in ("measure_dist", "measure_angle"):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Double-click: zoom to the clicked piece."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene().itemAt(scene_pos, self.transform())
            group = self._find_piece_group(item)
            if group is not None:
                self.fitToPiece(group.piece_id)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ── Keyboard events ────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() == Qt.Key.Key_Escape:
            self.setTool("select")
        elif event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.fitAll()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if self._tool == "pan":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self._tool in ("measure_dist", "measure_angle"):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(event)

    # ── Measure helpers ────────────────────────────────────────────

    def _handle_measure_click(self, scene_pos: QPointF) -> None:
        self._measure_points.append(scene_pos)

        if self._tool == "measure_dist" and len(self._measure_points) == 2:
            p1, p2 = self._measure_points
            dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            self._measure_overlay.set_distance(p1, p2, dist)
            self._measure_points.clear()

        elif self._tool == "measure_angle" and len(self._measure_points) == 3:
            p1, vertex, p3 = self._measure_points
            ax = p1.x() - vertex.x()
            ay = p1.y() - vertex.y()
            bx = p3.x() - vertex.x()
            by = p3.y() - vertex.y()
            dot = ax * bx + ay * by
            cross = ax * by - ay * bx
            angle = math.degrees(math.atan2(abs(cross), dot))
            self._measure_overlay.set_angle(p1, vertex, p3, angle)
            self._measure_points.clear()

    # ── Utilities ──────────────────────────────────────────────────

    @staticmethod
    def _find_piece_group(item) -> PieceGroup | None:
        """Walk up the item hierarchy to find a PieceGroup."""
        while item is not None:
            if isinstance(item, PieceGroup):
                return item
            item = item.parentItem()
        return None
