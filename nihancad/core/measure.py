"""Geometry measurement utilities for NihanCAD.

All coordinates and results in mm unless noted otherwise.
"""

from __future__ import annotations

import math


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.hypot(dx, dy)


def angle_3pt(p1: tuple[float, float], vertex: tuple[float, float], p3: tuple[float, float]) -> float:
    """Angle at *vertex* formed by rays to p1 and p3, in degrees [0, 360)."""
    ax, ay = p1[0] - vertex[0], p1[1] - vertex[1]
    bx, by = p3[0] - vertex[0], p3[1] - vertex[1]
    dot = ax * bx + ay * by
    cross = ax * by - ay * bx
    angle = math.atan2(abs(cross), dot)
    return math.degrees(angle)


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Signed area via the shoelace formula (mm²).  Returns absolute value."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_perimeter(points: list[tuple[float, float]]) -> float:
    """Perimeter of a closed polygon (mm)."""
    n = len(points)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += distance(points[i], points[j])
    return total


def point_to_segment(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> tuple[float, float, float]:
    """Shortest distance from point (px, py) to segment (x1,y1)-(x2,y2).

    Returns (distance, nearest_x, nearest_y).
    """
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        # Degenerate segment
        return distance((px, py), (x1, y1)), x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    nx = x1 + t * dx
    ny = y1 + t * dy
    d = math.hypot(px - nx, py - ny)
    return d, nx, ny


def centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Centroid of a closed polygon using the signed-area method.

    Falls back to arithmetic mean if area is near zero.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0
    if n < 3:
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        return cx, cy

    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = points[i][0] * points[j][1] - points[j][0] * points[i][1]
        signed_area += cross
        cx += (points[i][0] + points[j][0]) * cross
        cy += (points[i][1] + points[j][1]) * cross
    signed_area /= 2.0

    if abs(signed_area) < 1e-12:
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        return cx, cy

    cx /= 6.0 * signed_area
    cy /= 6.0 * signed_area
    return cx, cy
