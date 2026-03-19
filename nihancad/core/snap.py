"""Snap engine for NihanCAD — endpoint, midpoint, and nearest-point snapping."""

from __future__ import annotations

import math
from typing import Any

from nihancad.core.measure import distance, point_to_segment


class SnapResult:
    """Immutable snap hit."""

    __slots__ = ("type", "x", "y")

    def __init__(self, snap_type: str, x: float, y: float) -> None:
        self.type = snap_type  # 'endpoint' | 'midpoint' | 'nearest'
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"SnapResult({self.type!r}, {self.x:.2f}, {self.y:.2f})"


class SnapEngine:
    """Collects snap targets from pieces and resolves the best snap for a given cursor position."""

    def __init__(self) -> None:
        self.enabled: bool = True
        self.snap_radius: float = 15.0  # screen pixels
        self.types: dict[str, bool] = {
            "endpoint": True,
            "midpoint": True,
            "nearest": True,
        }

        self._endpoints: list[tuple[float, float]] = []
        self._midpoints: list[tuple[float, float]] = []
        self._edges: list[tuple[float, float, float, float]] = []  # (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, pieces: list[Any]) -> None:
        """Rebuild snap targets from a list of Piece objects.

        Collects:
          - endpoints: cutline vertices, seamline points, notch positions,
            refline start/end points
          - midpoints: cutline edge midpoints, refline midpoints
          - edges: cutline segments, seamline segments (for nearest snap)
        """
        self._endpoints.clear()
        self._midpoints.clear()
        self._edges.clear()

        for piece in pieces:
            # --- Cutline ---
            cutline = piece.cutline
            n = len(cutline)
            for i in range(n):
                self._endpoints.append(cutline[i])
                j = (i + 1) % n
                x1, y1 = cutline[i]
                x2, y2 = cutline[j]
                self._midpoints.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                self._edges.append((x1, y1, x2, y2))

            # --- Seamline ---
            seamline = piece.seamline
            for i, pt in enumerate(seamline):
                self._endpoints.append(pt)
                if i + 1 < len(seamline):
                    x1, y1 = pt
                    x2, y2 = seamline[i + 1]
                    self._edges.append((x1, y1, x2, y2))

            # --- Notches ---
            for notch in piece.notches:
                self._endpoints.append((notch.x, notch.y))

            # --- Ref lines ---
            for rl in piece.ref_lines:
                self._endpoints.append((rl.x1, rl.y1))
                self._endpoints.append((rl.x2, rl.y2))
                self._midpoints.append(((rl.x1 + rl.x2) / 2.0, (rl.y1 + rl.y2) / 2.0))

    # ------------------------------------------------------------------
    # Snap resolution
    # ------------------------------------------------------------------

    def find_snap(self, wx: float, wy: float, zoom: float) -> SnapResult | None:
        """Find the best snap target within radius for world coords (wx, wy).

        The snap radius is divided by *zoom* to convert from screen pixels to
        world units.

        Priority order: endpoint > midpoint > nearest.
        """
        if not self.enabled:
            return None

        threshold = self.snap_radius / max(zoom, 1e-9)
        best: SnapResult | None = None
        best_dist = threshold

        # --- Endpoint ---
        if self.types.get("endpoint", False):
            for ex, ey in self._endpoints:
                d = math.hypot(wx - ex, wy - ey)
                if d < best_dist:
                    best_dist = d
                    best = SnapResult("endpoint", ex, ey)

        # --- Midpoint (only wins if no endpoint was found) ---
        if self.types.get("midpoint", False) and (best is None or best.type != "endpoint"):
            mid_best_dist = best_dist if best is None else threshold
            for mx, my in self._midpoints:
                d = math.hypot(wx - mx, wy - my)
                if d < mid_best_dist:
                    mid_best_dist = d
                    best_dist = d
                    best = SnapResult("midpoint", mx, my)

        # --- Nearest (only if nothing better) ---
        if self.types.get("nearest", False) and best is None:
            near_best_dist = threshold
            for x1, y1, x2, y2 in self._edges:
                d, nx, ny = point_to_segment(wx, wy, x1, y1, x2, y2)
                if d < near_best_dist:
                    near_best_dist = d
                    best = SnapResult("nearest", nx, ny)

        return best
