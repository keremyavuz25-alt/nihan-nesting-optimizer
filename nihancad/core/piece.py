"""Piece model — converts parsed DXF blocks into structured garment pattern pieces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from nihancad.core.measure import (
    centroid as _centroid,
    distance,
    point_to_segment,
    polygon_area,
    polygon_perimeter,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Notch:
    x: float
    y: float
    label: str
    edge_angle: float = 0.0  # perpendicular to nearest cutline edge (degrees)


@dataclass
class Grainline:
    x: float
    y: float
    angle: float  # degrees, from DXF code 50
    length: float = 60.0  # mm, default display length


@dataclass
class RefLine:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class InternalLine:
    """Internal construction line (L8 open POLYLINE) — darts, pleats, stitching."""
    points: list[tuple[float, float]]


@dataclass
class LiningContour:
    """Lining cut contour (L8 closed POLYLINE) — pocket lining, facing."""
    points: list[tuple[float, float]]


@dataclass
class DrillPoint:
    """Drill/mark point (L13 POINT) — pocket corners, button positions."""
    x: float
    y: float


@dataclass
class Annotation:
    """Production annotation text (L8/L15 TEXT) — quantity, material notes."""
    x: float
    y: float
    text: str


@dataclass
class Piece:
    id: int
    name: str = ""
    size: str = ""
    material: str = ""
    quantity: int = 1
    annotation: str = ""
    cutline: list[tuple[float, float]] = field(default_factory=list)
    seamline: list[tuple[float, float]] = field(default_factory=list)
    grainlines: list[Grainline] = field(default_factory=list)
    notches: list[Notch] = field(default_factory=list)
    ref_lines: list[RefLine] = field(default_factory=list)
    internal_lines: list[InternalLine] = field(default_factory=list)
    lining_contours: list[LiningContour] = field(default_factory=list)
    drill_points: list[DrillPoint] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    area: float = 0.0       # mm²
    perimeter: float = 0.0  # mm
    width: float = 0.0
    height: float = 0.0
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    centroid: tuple[float, float] = (0.0, 0.0)
    color: str = "#E63946"


# 24-colour palette — pieces get colours round-robin.
COLORS: list[str] = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653",
    "#A8DADC", "#6D6875", "#B5838D", "#FFB4A2", "#95D5B2", "#D4A373",
    "#CDB4DB", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F0B27A",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEAMLINE_GAP_THRESHOLD = 50.0  # mm — if gap between consecutive L3 points exceeds this, break chain


def _parse_metadata(texts: list[dict[str, Any]]) -> dict[str, str]:
    """Extract KEY: VALUE pairs from layer-1 TEXT entities."""
    meta: dict[str, str] = {}
    for t in texts:
        raw: str = t.get("text", "")
        if ":" in raw:
            key, _, val = raw.partition(":")
            meta[key.strip().upper()] = val.strip()
    return meta


def _build_seamline(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Build a seamline from ordered layer-3 points, breaking at large gaps."""
    if len(points) < 2:
        return list(points)

    segments: list[list[tuple[float, float]]] = [[points[0]]]
    for i in range(1, len(points)):
        if distance(points[i - 1], points[i]) > _SEAMLINE_GAP_THRESHOLD:
            segments.append([points[i]])
        else:
            segments[-1].append(points[i])

    # Return the longest continuous chain
    best = max(segments, key=len)
    return best


def _build_grainlines(points: list[dict[str, Any]]) -> list[Grainline]:
    """Convert layer-4 points into Grainline objects.

    If two points share the same angle (within 1°), connect them as a line
    and use the inter-point distance as length. Otherwise single point + angle.
    """
    if not points:
        return []

    grainlines: list[Grainline] = []
    used: set[int] = set()

    for i in range(len(points)):
        if i in used:
            continue
        pi = points[i]
        ai = pi.get("angle", 0.0)
        matched = False
        for j in range(i + 1, len(points)):
            if j in used:
                continue
            pj = points[j]
            aj = pj.get("angle", 0.0)
            if abs(ai - aj) < 1.0:
                # Pair them
                mx = (pi["x"] + pj["x"]) / 2.0
                my = (pi["y"] + pj["y"]) / 2.0
                length = distance((pi["x"], pi["y"]), (pj["x"], pj["y"]))
                angle = math.degrees(math.atan2(pj["y"] - pi["y"], pj["x"] - pi["x"]))
                grainlines.append(Grainline(x=mx, y=my, angle=angle, length=length))
                used.add(i)
                used.add(j)
                matched = True
                break
        if not matched:
            grainlines.append(Grainline(x=pi["x"], y=pi["y"], angle=ai))
            used.add(i)

    return grainlines


def _match_notches(
    layer2_points: list[dict[str, Any]],
    layer2_texts: list[dict[str, Any]],
    cutline: list[tuple[float, float]],
) -> list[Notch]:
    """Pair layer-2 POINTs with the nearest following TEXT to form labelled notches.

    Then compute each notch's perpendicular angle relative to the nearest cutline edge.
    """
    notches: list[Notch] = []

    # Build ordered list of L2 entities preserving DXF order
    # The team-lead spec says TEXT follows its POINT; match each POINT with the next TEXT.
    point_queue: list[dict[str, Any]] = list(layer2_points)
    text_queue: list[dict[str, Any]] = list(layer2_texts)

    # Simple nearest-match: for each point, find the closest text
    for pt in point_queue:
        px, py = pt["x"], pt["y"]
        label = ""
        if text_queue:
            best_idx = 0
            best_dist = distance((px, py), (text_queue[0]["x"], text_queue[0]["y"]))
            for ti in range(1, len(text_queue)):
                d = distance((px, py), (text_queue[ti]["x"], text_queue[ti]["y"]))
                if d < best_dist:
                    best_dist = d
                    best_idx = ti
            label = text_queue[best_idx].get("text", "")
            text_queue.pop(best_idx)

        edge_angle = _notch_perpendicular(px, py, cutline)
        notches.append(Notch(x=px, y=py, label=label, edge_angle=edge_angle))

    return notches


def _notch_perpendicular(px: float, py: float, cutline: list[tuple[float, float]]) -> float:
    """Find the perpendicular angle (degrees) at the nearest cutline edge to (px, py)."""
    if len(cutline) < 2:
        return 0.0

    best_dist = float("inf")
    best_angle = 0.0
    n = len(cutline)
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = cutline[i]
        x2, y2 = cutline[j]
        d, _, _ = point_to_segment(px, py, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            edge_angle = math.atan2(y2 - y1, x2 - x1)
            best_angle = math.degrees(edge_angle + math.pi / 2.0)

    return best_angle


def _compute_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pieces(parsed_dxf: dict[str, Any]) -> list[Piece]:
    """Build Piece objects from a parsed DXF dict (output of DXFParser.parse).

    Skips blocks that have no cutline (layer-1 POLYLINE).
    """
    pieces: list[Piece] = []
    piece_id = 0

    for block in parsed_dxf.get("blocks", []):
        # Skip DXF model/paper-space blocks
        block_name: str = block.get("name", "")
        if block_name.startswith("*"):
            continue

        entities: list[dict] = block.get("entities", [])
        if not entities:
            continue

        # --- Classify entities by layer + type ---
        l1_polylines: list[dict] = []
        l1_texts: list[dict] = []
        l2_points: list[dict] = []
        l2_texts: list[dict] = []
        l3_points: list[tuple[float, float]] = []
        l4_points: list[dict] = []
        l7_lines: list[dict] = []
        l8_polylines: list[dict] = []
        l8_texts: list[dict] = []
        l8_lines: list[dict] = []
        l13_points: list[dict] = []
        l15_texts: list[dict] = []

        for ent in entities:
            etype = ent.get("type", "")
            layer = str(ent.get("layer", "0"))

            if etype == "POLYLINE" and layer == "1":
                l1_polylines.append(ent)
            elif etype == "TEXT" and layer == "1":
                l1_texts.append(ent)
            elif etype == "POINT" and layer == "2":
                l2_points.append(ent)
            elif etype == "TEXT" and layer == "2":
                l2_texts.append(ent)
            elif etype == "POINT" and layer == "3":
                l3_points.append((ent["x"], ent["y"]))
            elif etype == "POINT" and layer == "4":
                l4_points.append(ent)
            elif etype == "LINE" and layer == "7":
                l7_lines.append(ent)
            elif etype == "POLYLINE" and layer == "8":
                l8_polylines.append(ent)
            elif etype == "TEXT" and layer == "8":
                l8_texts.append(ent)
            elif etype == "LINE" and layer == "8":
                l8_lines.append(ent)
            elif etype == "POINT" and layer == "13":
                l13_points.append(ent)
            elif etype == "TEXT" and layer == "15":
                l15_texts.append(ent)

        # Must have at least one cutline
        if not l1_polylines:
            continue

        # Use the first (usually only) layer-1 POLYLINE as the cutline
        cutline = list(l1_polylines[0].get("vertices", []))

        # --- Metadata from layer-1 TEXTs ---
        meta = _parse_metadata(l1_texts)

        # L8: Internal lines (open polylines) and lining contours (closed polylines)
        internal_lines_list: list[InternalLine] = []
        lining_contours_list: list[LiningContour] = []
        for poly in l8_polylines:
            verts = list(poly.get("vertices", []))
            if len(verts) < 2:
                continue
            if poly.get("closed", False):
                lining_contours_list.append(LiningContour(points=verts))
            else:
                internal_lines_list.append(InternalLine(points=verts))
        # L8 LINEs are also internal lines (2-point segments)
        for ln in l8_lines:
            pts = [(ln["x1"], ln["y1"]), (ln["x2"], ln["y2"])]
            internal_lines_list.append(InternalLine(points=pts))

        # L8 + L15 TEXT: Annotations
        annotations_list: list[Annotation] = []
        for t in l8_texts:
            annotations_list.append(Annotation(x=t.get("x", 0), y=t.get("y", 0), text=t.get("text", "")))
        for t in l15_texts:
            annotations_list.append(Annotation(x=t.get("x", 0), y=t.get("y", 0), text=t.get("text", "")))

        # L13: Drill points
        drill_points_list: list[DrillPoint] = []
        for pt in l13_points:
            drill_points_list.append(DrillPoint(x=pt["x"], y=pt["y"]))

        # --- Build piece ---
        piece = Piece(
            id=piece_id,
            name=meta.get("PIECE NAME", meta.get("PIECENAME", block_name)),
            size=meta.get("SIZE", ""),
            material=meta.get("MATERIAL", ""),
            quantity=int(meta.get("QUANTITY", "1") or "1"),
            annotation=meta.get("ANNOTATION", ""),
            cutline=cutline,
            seamline=_build_seamline(l3_points),
            grainlines=_build_grainlines(l4_points),
            notches=_match_notches(l2_points, l2_texts, cutline),
            ref_lines=[
                RefLine(x1=ln["x1"], y1=ln["y1"], x2=ln["x2"], y2=ln["y2"])
                for ln in l7_lines
            ],
            internal_lines=internal_lines_list,
            lining_contours=lining_contours_list,
            drill_points=drill_points_list,
            annotations=annotations_list,
            color=COLORS[piece_id % len(COLORS)],
        )

        # --- Geometry ---
        if cutline:
            piece.area = polygon_area(cutline)
            piece.perimeter = polygon_perimeter(cutline)
            piece.bounds = _compute_bounds(cutline)
            minx, miny, maxx, maxy = piece.bounds
            piece.width = maxx - minx
            piece.height = maxy - miny
            piece.centroid = _centroid(cutline)

        pieces.append(piece)
        piece_id += 1

    return pieces
