"""QGraphicsItem subclasses for NihanCAD piece rendering.

Every visual aspect of a garment piece (cutline, seamline, grainline,
notches, reference lines, label text) is a separate QGraphicsItem.
They are grouped under a PieceGroup for unified selection/transforms.

Coordinate convention:
    Scene units = millimetres.  Y-axis is flipped at the *view* level
    (``scale(1, -1)``), so all items here use DXF coordinates directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from nihancad.graphics.layers import Layer

# ── Helpers ────────────────────────────────────────────────────────


def _make_pen(
    color: str,
    width: float,
    cosmetic: bool = True,
    dash: list[float] | None = None,
    opacity: float = 1.0,
) -> QPen:
    """Build a QPen from layer-style parameters."""
    qc = QColor(color)
    qc.setAlphaF(opacity)
    pen = QPen(qc, width)
    pen.setCosmetic(cosmetic)
    if dash:
        pen.setDashPattern(dash)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _path_from_points(points: list[tuple[float, float]], closed: bool = False) -> QPainterPath:
    """Create a QPainterPath from a list of (x, y) tuples."""
    path = QPainterPath()
    if not points:
        return path
    path.moveTo(QPointF(points[0][0], points[0][1]))
    for x, y in points[1:]:
        path.lineTo(QPointF(x, y))
    if closed:
        path.closeSubpath()
    return path


# ── PieceGroup ─────────────────────────────────────────────────────


class PieceGroup(QGraphicsItemGroup):
    """Groups every visual item of a single piece for unified selection.

    Attributes:
        piece_id: Unique piece identifier.
        piece_data: Reference to the source ``Piece`` dataclass.
    """

    def __init__(
        self,
        piece_id: int,
        piece_data: object | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self.piece_id: int = piece_id
        self.piece_data = piece_data

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)

        self._hovered = False

    # ── Selection / hover delegation to children ───────────────────

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            selected = bool(value)
            for child in self.childItems():
                if hasattr(child, 'set_selected'):
                    child.set_selected(selected)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):  # noqa: N802
        self._hovered = True
        for child in self.childItems():
            if hasattr(child, 'set_hovered'):
                child.set_hovered(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802
        self._hovered = False
        for child in self.childItems():
            if hasattr(child, 'set_hovered'):
                child.set_hovered(False)
        super().hoverLeaveEvent(event)

    @property
    def is_hovered(self) -> bool:
        return self._hovered


# ── CutlineItem ────────────────────────────────────────────────────


class CutlineItem(QGraphicsPathItem):
    """Closed polygon representing the cutting boundary of a piece.

    Renders with a solid stroke in the piece colour and a semi-transparent
    fill.  Stroke thickens on selection; fill brightens on hover.

    IMPORTANT: Never call setPen()/setBrush() from paint() — it triggers
    update() → paint() → infinite loop.  Selection/hover state is set
    externally via set_selected() / set_hovered().
    """

    def __init__(
        self,
        points: list[tuple[float, float]],
        color: str = "#e94560",
        layer: Layer | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        path = _path_from_points(points, closed=True)
        super().__init__(path, parent)

        self._base_color = color
        self._layer = layer
        self._build_pens_and_brushes()
        self.setPen(self._normal_pen)
        self.setBrush(self._normal_brush)

    def _build_pens_and_brushes(self) -> None:
        """Pre-build all pen/brush variants so paint() never mutates state."""
        lw = self._layer.line_width if self._layer else 2.0
        opacity = self._layer.opacity if self._layer else 0.8
        color = self._layer.color if self._layer else self._base_color

        self._normal_pen = _make_pen(color, lw, cosmetic=True, opacity=1.0)
        self._selected_pen = _make_pen(color, lw + 1.5, cosmetic=True, opacity=1.0)

        fill_normal = QColor(color)
        fill_normal.setAlphaF(opacity * 0.15)
        self._normal_brush = QBrush(fill_normal)

        fill_hover = QColor(color)
        fill_hover.setAlphaF(min(1.0, opacity * 0.25))
        self._hover_brush = QBrush(fill_hover)

    def set_selected(self, selected: bool) -> None:
        """Called from PieceGroup.itemChange(), NOT from paint()."""
        self.setPen(self._selected_pen if selected else self._normal_pen)

    def set_hovered(self, hovered: bool) -> None:
        """Called from PieceGroup.hoverEnter/LeaveEvent()."""
        self.setBrush(self._hover_brush if hovered else self._normal_brush)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        # Pure drawing — NO state mutation here.
        super().paint(painter, option, widget)

    def update_layer(self, layer: Layer) -> None:
        """Re-apply visual style from a changed layer."""
        self._layer = layer
        self._build_pens_and_brushes()
        self.setPen(self._normal_pen)
        self.setBrush(self._normal_brush)


# ── SeamlineItem ───────────────────────────────────────────────────


class SeamlineItem(QGraphicsPathItem):
    """Dashed line connecting seamline points.

    Gaps larger than ``GAP_THRESHOLD`` mm between consecutive points
    cause a ``moveTo`` instead of ``lineTo``, handling discontinuous
    seam data gracefully.
    """

    GAP_THRESHOLD: float = 50.0  # mm

    def __init__(
        self,
        points: list[tuple[float, float]],
        layer: Layer | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        path = self._build_path(points)
        super().__init__(path, parent)
        self._layer = layer
        self._apply_style()

    @classmethod
    def _build_path(cls, points: list[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        if not points:
            return path
        path.moveTo(QPointF(points[0][0], points[0][1]))
        for i in range(1, len(points)):
            px, py = points[i - 1]
            cx, cy = points[i]
            dist = math.hypot(cx - px, cy - py)
            if dist > cls.GAP_THRESHOLD:
                path.moveTo(QPointF(cx, cy))
            else:
                path.lineTo(QPointF(cx, cy))
        return path

    def _apply_style(self) -> None:
        color = self._layer.color if self._layer else "#00b4d8"
        lw = self._layer.line_width if self._layer else 1.0
        dash = self._layer.dash if self._layer else [6, 4]
        opacity = self._layer.opacity if self._layer else 0.9
        self.setPen(_make_pen(color, lw, cosmetic=True, dash=dash, opacity=opacity))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))

    def update_layer(self, layer: Layer) -> None:
        self._layer = layer
        self._apply_style()


# ── GrainlineItem ─────────────────────────────────────────────────


class GrainlineItem(QGraphicsItem):
    """Arrow showing fabric grain direction.

    Draws a line from ``(x, y)`` along ``angle`` for ``length`` mm,
    with a filled arrowhead at the end.
    """

    ARROW_WIDTH: float = 8.0   # mm half-width
    ARROW_LENGTH: float = 12.0  # mm

    def __init__(
        self,
        grainlines: list,
        layer: Layer | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._grainlines = grainlines  # list of Grainline dataclasses
        self._layer = layer
        self._build_geometry()

    def _build_geometry(self) -> None:
        """Pre-compute line endpoints and arrowhead polygons."""
        self._lines: list[tuple[QPointF, QPointF]] = []
        self._arrows: list[QPolygonF] = []
        self._bounds = QRectF()

        for gl in self._grainlines:
            x, y, angle_deg, length = gl.x, gl.y, gl.angle, gl.length
            rad = math.radians(angle_deg)
            dx = math.cos(rad) * length
            dy = math.sin(rad) * length
            p1 = QPointF(x, y)
            p2 = QPointF(x + dx, y + dy)
            self._lines.append((p1, p2))
            self._arrows.append(self._arrowhead(p1, p2))

        # Compute bounding rect from all points.
        all_pts: list[QPointF] = []
        for p1, p2 in self._lines:
            all_pts.extend([p1, p2])
        for poly in self._arrows:
            for i in range(poly.count()):
                all_pts.append(poly.at(i))
        if all_pts:
            xs = [p.x() for p in all_pts]
            ys = [p.y() for p in all_pts]
            margin = 2.0
            self._bounds = QRectF(
                min(xs) - margin, min(ys) - margin,
                (max(xs) - min(xs)) + 2 * margin,
                (max(ys) - min(ys)) + 2 * margin,
            )

    def _arrowhead(self, p1: QPointF, p2: QPointF) -> QPolygonF:
        """Filled triangle at *p2* pointing away from *p1*."""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return QPolygonF()

        ux, uy = dx / length, dy / length  # unit direction
        nx, ny = -uy, ux  # normal

        tip = p2
        base_center = QPointF(
            p2.x() - ux * self.ARROW_LENGTH,
            p2.y() - uy * self.ARROW_LENGTH,
        )
        left = QPointF(
            base_center.x() + nx * self.ARROW_WIDTH / 2,
            base_center.y() + ny * self.ARROW_WIDTH / 2,
        )
        right = QPointF(
            base_center.x() - nx * self.ARROW_WIDTH / 2,
            base_center.y() - ny * self.ARROW_WIDTH / 2,
        )
        return QPolygonF([tip, left, right, tip])

    # ── QGraphicsItem interface ────────────────────────────────────

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        color = self._layer.color if self._layer else "#48bb78"
        lw = self._layer.line_width if self._layer else 1.5
        pen = _make_pen(color, lw, cosmetic=True)
        painter.setPen(pen)

        qc = QColor(color)
        painter.setBrush(QBrush(qc))

        for p1, p2 in self._lines:
            painter.drawLine(p1, p2)
        for poly in self._arrows:
            if poly.count() > 0:
                painter.drawPolygon(poly)

    def update_layer(self, layer: Layer) -> None:
        self._layer = layer
        self.update()


# ── NotchItem ──────────────────────────────────────────────────────


class NotchItem(QGraphicsItem):
    """Small perpendicular tick marks on the cutline edge, with labels.

    Each notch is drawn as a short line (5 mm) perpendicular to the
    cutline at the notch position, plus a small text label offset
    from the tick.
    """

    TICK_LENGTH: float = 5.0  # mm

    def __init__(
        self,
        notches: list,
        layer: Layer | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._notches = notches  # list of Notch dataclasses
        self._layer = layer
        self._build_geometry()

    def _build_geometry(self) -> None:
        self._ticks: list[tuple[QPointF, QPointF]] = []
        self._labels: list[tuple[QPointF, str]] = []
        self._bounds = QRectF()

        all_pts: list[QPointF] = []
        for n in self._notches:
            # Perpendicular direction to edge_angle
            perp_rad = math.radians(n.edge_angle + 90)
            dx = math.cos(perp_rad) * self.TICK_LENGTH
            dy = math.sin(perp_rad) * self.TICK_LENGTH

            p1 = QPointF(n.x, n.y)
            p2 = QPointF(n.x + dx, n.y + dy)
            self._ticks.append((p1, p2))
            all_pts.extend([p1, p2])

            if n.label:
                label_pt = QPointF(n.x + dx * 1.5, n.y + dy * 1.5)
                self._labels.append((label_pt, n.label))
                all_pts.append(label_pt)

        if all_pts:
            xs = [p.x() for p in all_pts]
            ys = [p.y() for p in all_pts]
            margin = 10.0
            self._bounds = QRectF(
                min(xs) - margin, min(ys) - margin,
                (max(xs) - min(xs)) + 2 * margin,
                (max(ys) - min(ys)) + 2 * margin,
            )

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        color = self._layer.color if self._layer else "#ed8936"
        lw = self._layer.line_width if self._layer else 1.0
        pen = _make_pen(color, lw, cosmetic=True)
        painter.setPen(pen)

        for p1, p2 in self._ticks:
            painter.drawLine(p1, p2)

        # Draw labels — need to invert Y for readable text since view is flipped.
        font = QFont("Segoe UI", 7)
        painter.setFont(font)
        for pt, label in self._labels:
            painter.save()
            painter.translate(pt)
            painter.scale(1, -1)  # un-flip text
            painter.drawText(QPointF(0, 0), label)
            painter.restore()

    def update_layer(self, layer: Layer) -> None:
        self._layer = layer
        self.update()


# ── RefLineItem ────────────────────────────────────────────────────


class RefLineItem(QGraphicsLineItem):
    """Thin dashed reference/guide line."""

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        layer: Layer | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(x1, y1, x2, y2, parent)
        self._layer = layer
        self._apply_style()

    def _apply_style(self) -> None:
        color = self._layer.color if self._layer else "#888888"
        lw = self._layer.line_width if self._layer else 0.5
        dash = self._layer.dash if self._layer else [4, 4]
        opacity = self._layer.opacity if self._layer else 0.5
        self.setPen(_make_pen(color, lw, cosmetic=True, dash=dash, opacity=opacity))

    def update_layer(self, layer: Layer) -> None:
        self._layer = layer
        self._apply_style()


# ── InternalLineItem ──────────────────────────────────────────────


class InternalLineItem(QGraphicsPathItem):
    """Internal construction lines (darts, pleats, stitching) from L8 open POLYLINE."""

    def __init__(self, internal_lines: list, layer=None, parent=None):
        path = QPainterPath()
        for il in internal_lines:
            pts = il.points
            if len(pts) < 2:
                continue
            path.moveTo(QPointF(pts[0][0], pts[0][1]))
            for x, y in pts[1:]:
                path.lineTo(QPointF(x, y))
        super().__init__(path, parent)
        self._layer = layer
        self._apply_style()

    def _apply_style(self):
        color = self._layer.color if self._layer else "#c792ea"
        lw = self._layer.line_width if self._layer else 1.0
        dash = self._layer.dash if self._layer else [4, 3]
        opacity = self._layer.opacity if self._layer else 0.8
        self.setPen(_make_pen(color, lw, cosmetic=True, dash=dash, opacity=opacity))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))

    def update_layer(self, layer):
        self._layer = layer
        self._apply_style()


# ── LiningItem ────────────────────────────────────────────────────


class LiningItem(QGraphicsPathItem):
    """Lining cut contours (pocket lining, facing) from L8 closed POLYLINE."""

    def __init__(self, lining_contours: list, layer=None, parent=None):
        path = QPainterPath()
        for lc in lining_contours:
            pts = lc.points
            if len(pts) < 3:
                continue
            path.moveTo(QPointF(pts[0][0], pts[0][1]))
            for x, y in pts[1:]:
                path.lineTo(QPointF(x, y))
            path.closeSubpath()
        super().__init__(path, parent)
        self._layer = layer
        self._apply_style()

    def _apply_style(self):
        color = self._layer.color if self._layer else "#82aaff"
        lw = self._layer.line_width if self._layer else 1.5
        dash = self._layer.dash if self._layer else [8, 4]
        opacity = self._layer.opacity if self._layer else 0.7
        self.setPen(_make_pen(color, lw, cosmetic=True, dash=dash, opacity=opacity))
        fill = QColor(color)
        fill.setAlphaF(opacity * 0.08)
        self.setBrush(QBrush(fill))

    def update_layer(self, layer):
        self._layer = layer
        self._apply_style()


# ── DrillItem ─────────────────────────────────────────────────────


class DrillItem(QGraphicsItem):
    """Drill/mark points (pocket corners, button positions) from L13 POINT."""

    CROSS_SIZE: float = 4.0  # mm half-extent

    def __init__(self, drill_points: list, layer=None, parent=None):
        super().__init__(parent)
        self._drill_points = drill_points
        self._layer = layer
        self._bounds = QRectF()
        self._build_geometry()

    def _build_geometry(self):
        if not self._drill_points:
            self._bounds = QRectF()
            return
        xs = [dp.x for dp in self._drill_points]
        ys = [dp.y for dp in self._drill_points]
        margin = self.CROSS_SIZE + 2
        self._bounds = QRectF(
            min(xs) - margin, min(ys) - margin,
            (max(xs) - min(xs)) + 2 * margin,
            (max(ys) - min(ys)) + 2 * margin,
        )

    def boundingRect(self):
        return self._bounds

    def paint(self, painter, option, widget=None):
        color = self._layer.color if self._layer else "#ffcb6b"
        lw = self._layer.line_width if self._layer else 1.0
        pen = _make_pen(color, lw, cosmetic=True)
        painter.setPen(pen)
        s = self.CROSS_SIZE
        for dp in self._drill_points:
            # Draw cross mark
            painter.drawLine(QPointF(dp.x - s, dp.y), QPointF(dp.x + s, dp.y))
            painter.drawLine(QPointF(dp.x, dp.y - s), QPointF(dp.x, dp.y + s))
            # Draw small circle
            painter.drawEllipse(QPointF(dp.x, dp.y), s * 0.6, s * 0.6)

    def update_layer(self, layer):
        self._layer = layer
        self.update()


# ── AnnotationItem ────────────────────────────────────────────────


class AnnotationItem(QGraphicsItem):
    """Production annotation text from L8/L15 TEXT — quantity, material notes."""

    def __init__(self, annotations: list, layer=None, parent=None):
        super().__init__(parent)
        self._annotations = annotations
        self._layer = layer
        self._bounds = QRectF()
        self._build_geometry()

    def _build_geometry(self):
        if not self._annotations:
            self._bounds = QRectF()
            return
        xs = [a.x for a in self._annotations]
        ys = [a.y for a in self._annotations]
        margin = 20.0
        self._bounds = QRectF(
            min(xs) - margin, min(ys) - margin,
            (max(xs) - min(xs)) + 2 * margin,
            (max(ys) - min(ys)) + 2 * margin,
        )

    def boundingRect(self):
        return self._bounds

    def paint(self, painter, option, widget=None):
        color = self._layer.color if self._layer else "#c3e88d"
        painter.setPen(QColor(color))
        font = QFont("Segoe UI", 8)
        font.setItalic(True)
        painter.setFont(font)
        for a in self._annotations:
            painter.save()
            painter.translate(QPointF(a.x, a.y))
            painter.scale(1, -1)  # un-flip for readable text
            painter.drawText(QPointF(0, 0), a.text)
            painter.restore()

    def update_layer(self, layer):
        self._layer = layer
        self.update()


# ── PieceTextItem ──────────────────────────────────────────────────


class PieceTextItem(QGraphicsSimpleTextItem):
    """Piece name rendered at the centroid.

    NOTE: ItemIgnoresTransformations is NOT used here because it breaks
    QGraphicsItemGroup.boundingRect() when this item is a child.
    Instead, the text is added as a top-level scene item (not in the
    PieceGroup) and positioned at the piece centroid.  A local
    ``scale(1, -1)`` counteracts the view's Y-flip.
    """

    def __init__(
        self,
        text: str,
        cx: float,
        cy: float,
        color: str = "#aaaaaa",
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(text, parent)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setPos(cx, cy)

        # Un-flip Y so text reads correctly in the flipped view.
        self.setTransform(QTransform.fromScale(1, -1))

        font = QFont("Segoe UI", 10)
        font.setBold(True)
        self.setFont(font)

        qc = QColor(color)
        self.setBrush(QBrush(qc))

    def update_layer(self, layer: Layer) -> None:
        if layer:
            self.setBrush(QBrush(QColor(layer.color)))


# ── MeasureOverlay ─────────────────────────────────────────────────


class MeasureOverlay(QGraphicsItem):
    """Measurement visualisation drawn on top of everything.

    Supports two modes:
        * **distance** — line between two points, dimension text at midpoint.
        * **angle** — arc between three points, degree text near vertex.
    """

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.setZValue(1000)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

        self._mode: str = "distance"  # 'distance' or 'angle'
        self._points: list[QPointF] = []
        self._text: str = ""
        self._bounds = QRectF()

    def set_distance(self, p1: QPointF, p2: QPointF, dist_mm: float) -> None:
        """Configure the overlay to show a distance measurement."""
        self._mode = "distance"
        self._points = [p1, p2]
        self._text = f"{dist_mm:.1f} mm"
        self._recompute_bounds()
        self.update()

    def set_angle(self, p1: QPointF, vertex: QPointF, p3: QPointF, degrees: float) -> None:
        """Configure the overlay to show an angle measurement."""
        self._mode = "angle"
        self._points = [p1, vertex, p3]
        self._text = f"{degrees:.1f}°"
        self._recompute_bounds()
        self.update()

    def clear(self) -> None:
        self._points.clear()
        self._text = ""
        self._bounds = QRectF()
        self.update()

    def _recompute_bounds(self) -> None:
        if not self._points:
            self._bounds = QRectF()
            return
        xs = [p.x() for p in self._points]
        ys = [p.y() for p in self._points]
        margin = 20.0
        self._bounds = QRectF(
            min(xs) - margin, min(ys) - margin,
            (max(xs) - min(xs)) + 2 * margin,
            (max(ys) - min(ys)) + 2 * margin,
        )

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        if not self._points:
            return

        pen = QPen(QColor("#ff4444"), 1.0)
        pen.setCosmetic(True)
        pen.setDashPattern([6, 3])
        painter.setPen(pen)

        if self._mode == "distance" and len(self._points) == 2:
            p1, p2 = self._points
            painter.drawLine(p1, p2)
            mid = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
            self._draw_label(painter, mid, self._text)

        elif self._mode == "angle" and len(self._points) == 3:
            p1, vertex, p3 = self._points
            painter.drawLine(vertex, p1)
            painter.drawLine(vertex, p3)
            # Draw arc
            radius = 15.0
            start_angle = math.degrees(math.atan2(p1.y() - vertex.y(), p1.x() - vertex.x()))
            end_angle = math.degrees(math.atan2(p3.y() - vertex.y(), p3.x() - vertex.x()))
            span = end_angle - start_angle
            if span < 0:
                span += 360
            arc_rect = QRectF(vertex.x() - radius, vertex.y() - radius, 2 * radius, 2 * radius)
            painter.drawArc(arc_rect, int(-start_angle * 16), int(-span * 16))
            # Label offset from vertex
            label_angle = math.radians(start_angle + span / 2)
            label_pt = QPointF(
                vertex.x() + math.cos(label_angle) * radius * 1.8,
                vertex.y() + math.sin(label_angle) * radius * 1.8,
            )
            self._draw_label(painter, label_pt, self._text)

    def _draw_label(self, painter: QPainter, pos: QPointF, text: str) -> None:
        """Draw text with a white background at *pos* (un-flipped)."""
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.save()
        painter.translate(pos)
        painter.scale(1, -1)  # un-flip for readability

        fm = painter.fontMetrics()
        rect = fm.boundingRect(text)
        bg = QRectF(rect).adjusted(-3, -2, 3, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
        painter.drawRoundedRect(bg, 2, 2)

        painter.setPen(QColor("#ff4444"))
        painter.drawText(QPointF(0, 0), text)
        painter.restore()


# ── SnapIndicator ──────────────────────────────────────────────────


class SnapIndicator(QGraphicsItem):
    """Visual indicator at a snap point.

    Shape varies by snap type:
        * endpoint — diamond
        * midpoint — triangle
        * nearest — circle

    Rendered at fixed screen size (``ItemIgnoresTransformations``).
    Always on top (high Z value).
    """

    SIZE: float = 4.0  # screen pixels (half-extent)

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(2000)
        self.setVisible(False)

        self._snap_type: str = "endpoint"

    def show_at(self, x: float, y: float, snap_type: str = "endpoint") -> None:
        """Display the indicator at scene position *(x, y)*."""
        self._snap_type = snap_type
        self.setPos(x, y)
        self.setVisible(True)
        self.update()

    def hide(self) -> None:
        self.setVisible(False)

    def boundingRect(self) -> QRectF:  # noqa: N802
        s = self.SIZE + 1
        return QRectF(-s, -s, 2 * s, 2 * s)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        color = QColor("#FFD700")
        pen = QPen(color, 1.5)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))

        s = self.SIZE
        if self._snap_type == "endpoint":
            # Diamond
            poly = QPolygonF([
                QPointF(0, -s),
                QPointF(s, 0),
                QPointF(0, s),
                QPointF(-s, 0),
            ])
            painter.drawPolygon(poly)

        elif self._snap_type == "midpoint":
            # Triangle
            poly = QPolygonF([
                QPointF(0, -s),
                QPointF(s, s),
                QPointF(-s, s),
            ])
            painter.drawPolygon(poly)

        else:  # 'nearest' or fallback
            painter.drawEllipse(QPointF(0, 0), s, s)
