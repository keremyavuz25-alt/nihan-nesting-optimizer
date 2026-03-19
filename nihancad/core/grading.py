"""Grading engine for GEMX pattern files.

Extracts multiple graded sizes from expanded GEMX files by tracking
piece-name repetitions in the rp2.rp binary stream.

GEMX file structure (per piece, per size):
    PieceName → ÇİZİM → cutline contour(s) + notch/grain/drill
    PieceName (repeat) → next size block

Each name repetition represents a different graded size of the same piece.
Contours between consecutive name occurrences belong to that size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from nihancad.core.piece import (
    DrillPoint, Grainline, Notch, Piece, COLORS,
)
from nihancad.core.measure import (
    centroid as _centroid,
    polygon_area,
    polygon_perimeter,
)


# Production notes and meta labels — not piece names
_SKIP_NAMES = {
    "N/A", "Default", "Piese libere",
    "KUMAS", "ASTAR", "TELA", "SEZON",
}

_SKIP_KEYWORDS = {"ADET", "KESILSIN", "TELALI", "TELASIZ", "IPTAL"}


def _is_real_piece_name(name: str) -> bool:
    """True if *name* is a garment piece identifier, not a meta/production label."""
    name = name.strip()
    if not name or len(name) < 2:
        return False
    if name in _SKIP_NAMES:
        return False
    if name.isdigit():
        return False
    # ÇİZİM (drawing layer marker)
    if name in ("ÇİZİM", "ÇÖZÜM", "\xc7\xd6Z\xdcM"):
        return False
    # Production notes
    for kw in _SKIP_KEYWORDS:
        if kw in name:
            return False
    # Names with spaces are usually production notes
    if " " in name.strip() and not name.startswith("N/"):
        return False
    return True


@dataclass
class GradedPiece:
    """One piece template with multiple graded sizes."""
    name: str
    sizes: list[SizeInstance] = field(default_factory=list)


@dataclass
class SizeInstance:
    """One graded size of a piece — cutline + layer data."""
    size_index: int          # 0-based index within the grade set
    size_label: str          # e.g. "38", "1", "" if unknown
    cutline: list[tuple[float, float]] = field(default_factory=list)
    grainlines: list[Grainline] = field(default_factory=list)
    drill_points: list[DrillPoint] = field(default_factory=list)
    notches: list[Notch] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


def extract_graded_pieces(
    all_contours: list[dict],
    names: list[tuple[int, str]],
    tessellate_fn,
    size_labels: list[str | int] | None = None,
) -> list[GradedPiece]:
    """Extract graded pieces from GEMX contour + name data.

    Args:
        all_contours: ALL v<ui8b13> contours (from _extract_all_gemx_contours).
        names: Piece names with offsets (from _extract_piece_names).
        tessellate_fn: Bezier tessellation function (GemParser._tessellate_bezier_contour).
        size_labels: Optional real size labels (e.g. [38, 40, 42, 44] from ölçü tablosu).

    Returns:
        List of GradedPiece, each containing multiple SizeInstance objects.
    """
    # Sort contours by offset
    sorted_contours = sorted(all_contours, key=lambda c: c["offset"])

    # Build event stream: names + contours in file order
    events: list[tuple[int, str, object]] = []
    for off, name in names:
        events.append((off, "name", name.strip()))
    for c in sorted_contours:
        events.append((c["offset"], "contour", c))
    events.sort(key=lambda e: e[0])

    # Walk events, grouping contours by piece-name blocks
    # Each block = one size of one piece
    blocks: list[tuple[str, list[dict]]] = []  # (piece_name, contours)
    current_piece: str | None = None
    current_contours: list[dict] = []
    last_was_cizim = False

    for off, typ, val in events:
        if typ == "name":
            name = val
            if not _is_real_piece_name(name):
                # Check if it's ÇİZİM marker
                if name in ("ÇİZİM", "ÇÖZÜM", "\xc7\xd6Z\xdcM"):
                    last_was_cizim = True
                continue

            if name == current_piece:
                # Same piece name again → close current block, start new size
                if current_contours:
                    blocks.append((current_piece, list(current_contours)))
                    current_contours = []
            else:
                # New piece → close previous block
                if current_piece and current_contours:
                    blocks.append((current_piece, list(current_contours)))
                current_piece = name
                current_contours = []
            last_was_cizim = False

        elif typ == "contour" and current_piece:
            current_contours.append(val)

    # Close final block
    if current_piece and current_contours:
        blocks.append((current_piece, list(current_contours)))

    # Group blocks by piece name → GradedPiece
    from collections import OrderedDict
    piece_map: OrderedDict[str, list[tuple[str, list[dict]]]] = OrderedDict()
    for name, contours in blocks:
        piece_map.setdefault(name, []).append((name, contours))

    # Build GradedPiece objects
    graded_pieces: list[GradedPiece] = []

    for piece_name, size_blocks in piece_map.items():
        gp = GradedPiece(name=piece_name)

        for size_idx, (_, contours) in enumerate(size_blocks):
            si = _classify_size_contours(contours, tessellate_fn, size_idx)
            if si is None:
                continue

            # Assign size label
            if size_labels and size_idx < len(size_labels):
                si.size_label = str(size_labels[size_idx])

            gp.sizes.append(si)

        if gp.sizes:
            graded_pieces.append(gp)

    return graded_pieces


def _classify_size_contours(
    contours: list[dict],
    tessellate_fn,
    size_index: int,
) -> SizeInstance | None:
    """Classify contours within one size block into cutline/grain/drill/notch."""
    cutline_c = None
    grainline_c = None
    drill_pts: list[tuple[float, float]] = []
    notch_pts: list[tuple[float, float, float]] = []

    for c in contours:
        w, h, n = c["width"], c["height"], c["count"]
        maxdim = max(w, h)

        if maxdim > 50 and n > 5:
            # Cutline candidate — take the largest
            if cutline_c is None or (w * h) > (cutline_c["width"] * cutline_c["height"]):
                cutline_c = c
        elif n == 2 and maxdim > 30 and min(w, h) < 5:
            grainline_c = c
        elif n == 1 and maxdim < 0.1:
            vx, vy, _ = c["vertices"][0]
            drill_pts.append((vx, vy))
        elif n == 2 and maxdim < 15:
            v0, v1 = c["vertices"][0], c["vertices"][1]
            mx = (v0[0] + v1[0]) / 2
            my = (v0[1] + v1[1]) / 2
            angle = math.degrees(math.atan2(v1[1] - v0[1], v1[0] - v0[0]))
            notch_pts.append((mx, my, angle))

    if cutline_c is None:
        return None

    cutline_pts = tessellate_fn(cutline_c["vertices"])
    if len(cutline_pts) < 3:
        return None

    # Build grainline
    grainlines: list[Grainline] = []
    if grainline_c:
        v0, v1 = grainline_c["vertices"][0], grainline_c["vertices"][1]
        gx = (v0[0] + v1[0]) / 2
        gy = (v0[1] + v1[1]) / 2
        g_angle = math.degrees(math.atan2(v1[1] - v0[1], v1[0] - v0[0]))
        g_length = math.hypot(v1[0] - v0[0], v1[1] - v0[1])
        grainlines.append(Grainline(x=gx, y=gy, angle=g_angle, length=g_length))

    # Deduplicate notch pairs
    notches: list[Notch] = []
    used: set[int] = set()
    for i, (nx, ny, na) in enumerate(notch_pts):
        if i in used:
            continue
        for j in range(i + 1, len(notch_pts)):
            if j in used:
                continue
            ox, oy, _ = notch_pts[j]
            if math.hypot(nx - ox, ny - oy) < 1.0:
                used.add(j)
                break
        notches.append(Notch(x=nx, y=ny, label="", edge_angle=na))

    return SizeInstance(
        size_index=size_index,
        size_label="",
        cutline=cutline_pts,
        grainlines=grainlines,
        drill_points=[DrillPoint(x=dx, y=dy) for dx, dy in drill_pts],
        notches=notches,
        width=cutline_c["width"],
        height=cutline_c["height"],
    )


def graded_pieces_to_flat(
    graded: list[GradedPiece],
) -> list[Piece]:
    """Convert GradedPiece list to flat Piece list for rendering.

    Each size of each piece becomes a separate Piece object.
    """
    pieces: list[Piece] = []
    piece_id = 0

    for gp in graded:
        num_sizes = len(gp.sizes)
        for si in gp.sizes:
            if num_sizes > 1:
                label = si.size_label or str(si.size_index + 1)
                display_name = f"{gp.name} [{label}]"
            else:
                display_name = gp.name

            # Normalize cutline
            if not si.cutline:
                continue
            xs = [p[0] for p in si.cutline]
            ys = [p[1] for p in si.cutline]
            min_x, min_y = min(xs), min(ys)
            normalized = [(x - min_x, y - min_y) for x, y in si.cutline]

            piece = Piece(
                id=piece_id,
                name=display_name,
                size=si.size_label,
                cutline=normalized,
                grainlines=si.grainlines,
                drill_points=si.drill_points,
                notches=si.notches,
                color=COLORS[piece_id % len(COLORS)],
            )

            piece.width = max(xs) - min_x
            piece.height = max(ys) - min_y
            piece.bounds = (0.0, 0.0, piece.width, piece.height)
            piece.area = polygon_area(normalized)
            piece.perimeter = polygon_perimeter(normalized)
            piece.centroid = _centroid(normalized)

            pieces.append(piece)
            piece_id += 1

    return pieces
