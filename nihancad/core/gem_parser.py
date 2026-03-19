"""GEM / GEMX binary parser for Gemini CAD Systems pattern files.

Handles two proprietary formats:
  - GEMX (newer): ZIP containing models/rp2.rp with TLV-tagged Bezier geometry
  - GEM  (older): ZIP containing an inner .gem binary with float64 vertex data

Both formats are reverse-engineered from hex analysis of production files.
"""

from __future__ import annotations

import math
import struct
import zipfile
from pathlib import Path
from statistics import median

from nihancad.core.measure import (
    centroid as _centroid,
    polygon_area,
    polygon_perimeter,
)
from nihancad.core.piece import COLORS, DrillPoint, Grainline, Notch, Piece


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BEZIER_STEPS = 16          # tessellation resolution per cubic segment
_MIN_PIECE_BBOX_MM = 30.0   # filter out notch symbols / markers
_GEM_HEADER_SIZE = 256      # approximate global header in inner .gem
_OUTLIER_SIGMA = 2.5        # MAD-based outlier threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GemParser:
    """Parse GEM and GEMX files from Gemini CAD Systems."""

    def parse(
        self,
        filepath: str,
        size_labels: list[str | int] | None = None,
    ) -> list[Piece]:
        """Auto-detect GEM vs GEMX and parse.

        Args:
            filepath: Path to .gem or .gemx file.
            size_labels: Optional real size labels from ölçü tablosu
                (e.g. [38, 40, 42, 44] or [1, 2, 3]).
                Mapped to graded sizes in order.

        Returns list of Piece objects compatible with DXF-parsed pieces.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            zf = zipfile.ZipFile(filepath, "r")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Not a valid ZIP archive: {filepath}") from exc

        try:
            names = zf.namelist()
            if any(n == "models/rp2.rp" or n.endswith("/rp2.rp") for n in names):
                return self._parse_gemx(zf, size_labels=size_labels)
            # Legacy GEM: look for an inner .gem binary
            gem_inner = [n for n in names if n.lower().endswith(".gem")]
            if gem_inner:
                return self._parse_gem(zf, gem_inner[0])
            raise ValueError(
                f"Unrecognized archive layout — no rp2.rp or inner .gem found in {filepath}"
            )
        finally:
            zf.close()

    # -----------------------------------------------------------------------
    # GEMX format
    # -----------------------------------------------------------------------

    def _parse_gemx(
        self,
        zf: zipfile.ZipFile,
        size_labels: list[str | int] | None = None,
    ) -> list[Piece]:
        """Parse GEMX format (models/rp2.rp TLV tree with Bezier splines).

        Uses the grading engine to extract multiple sizes per piece.
        Each piece-name repetition in the file = a different graded size.

        Args:
            zf: Open ZipFile for the GEMX.
            size_labels: Optional real size labels from ölçü tablosu
                (e.g. [38, 40, 42, 44]).  Mapped to sizes in order.
        """
        from nihancad.core.grading import extract_graded_pieces, graded_pieces_to_flat

        rp2_name = next(
            (n for n in zf.namelist() if n == "models/rp2.rp" or n.endswith("/rp2.rp")),
            None,
        )
        if rp2_name is None:
            return []

        data = zf.read(rp2_name)

        # 1) Extract ALL v<ui8b13> geometry sections (no bbox filter)
        all_contours = self._extract_all_gemx_contours(data)

        # 2) Extract piece names from UTF-16LE strings
        piece_names = self._extract_piece_names(data)

        # 3) Use grading engine to extract all sizes
        graded = extract_graded_pieces(
            all_contours,
            piece_names,
            self._tessellate_bezier_contour,
            size_labels=size_labels,
        )

        # 4) Flatten to Piece list
        return graded_pieces_to_flat(graded)

    def _extract_all_gemx_contours(self, data: bytes) -> list[dict]:
        """Find ALL v<ui8b13> geometry sections — no bbox filter.

        Returns list of dicts with keys: offset, vertices, width, height, count.
        Each vertex is (x_mm, y_mm, type_byte).
        """
        tag = b"v<ui8b13>"
        tag_len = len(tag)  # 9
        contours: list[dict] = []
        pos = 0

        while True:
            pos = data.find(tag, pos)
            if pos == -1:
                break

            if pos + tag_len + 4 > len(data):
                break
            body_start = pos + tag_len + 4

            if body_start + 16 > len(data):
                pos += 1
                continue

            version, fmt, count, dsize = struct.unpack_from("<IIII", data, body_start)

            if version != 0 or fmt != 4 or count == 0 or count > 50000:
                pos += 1
                continue
            if dsize != count * 13:
                pos += 1
                continue

            rec_start = body_start + 16
            if rec_start + count * 13 > len(data):
                pos += 1
                continue

            vertices: list[tuple[float, float, int]] = []
            for i in range(count):
                off = rec_start + i * 13
                x_mic, y_mic = struct.unpack_from("<ii", data, off)
                type_byte = data[off + 10]
                vertices.append((x_mic / 1000.0, y_mic / 1000.0, type_byte))

            if not vertices:
                pos += 1
                continue

            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)

            contours.append({
                "offset": pos,
                "vertices": vertices,
                "width": w,
                "height": h,
                "count": count,
            })

            pos += 1

        return contours

    def _extract_gemx_contours(self, data: bytes) -> list[dict]:
        """Legacy method: extract only large contours (cutlines).

        Kept for backwards compatibility.
        """
        all_c = self._extract_all_gemx_contours(data)
        return [
            c for c in all_c
            if c["width"] >= _MIN_PIECE_BBOX_MM and c["height"] >= _MIN_PIECE_BBOX_MM
        ]

    def _extract_piece_names(self, data: bytes) -> list[tuple[int, str]]:
        """Extract piece names from rp2.rp UTF-16LE strings.

        Format: [u32 charCount][u8 marker=1][UTF-16LE chars]
        Returns sorted list of (offset, name).
        """
        results: list[tuple[int, str]] = []
        # Known labels to skip (material/meta, not piece names)
        _META_LABELS = {
            "KUMAS", "ASTAR", "TELA", "SEZON",
            "Piece size", "Piece name",
        }

        end = len(data) - 10
        offset = 0
        while offset < end:
            char_count = struct.unpack_from("<I", data, offset)[0]
            if not (2 <= char_count <= 25):
                offset += 1
                continue

            if data[offset + 4] != 1:
                offset += 1
                continue

            name_start = offset + 5
            name_end = name_start + char_count * 2
            if name_end > len(data):
                offset += 1
                continue

            try:
                name = data[name_start:name_end].decode("utf-16-le")
            except (UnicodeDecodeError, ValueError):
                offset += 1
                continue

            # Valid piece name: uppercase letters, digits, limited punctuation
            if (
                name
                and len(name) >= 2
                and all(
                    c in "ABCDEFGHIJKLMNOPQRSTUVWXYZÇĞİÖŞÜ0123456789 .-_/"
                    for c in name
                )
                and name not in _META_LABELS
            ):
                results.append((offset, name))
                offset = name_end  # skip past this name
            else:
                offset += 1

        return results

    def _match_gemx_pieces(
        self,
        contours: list[dict],
        names: list[tuple[int, str]],
        data: bytes,
    ) -> list[Piece]:
        """Match contours to piece names and build Piece objects with full layer data.

        Grading-aware: GEMX files may contain multiple sizes of the same piece
        as separate expanded contours.  These are detected by spatial clustering
        of cutline contours — contours at similar locations but different sizes
        belong to the same piece template at different grades.

        Each graded size becomes a separate Piece with a ``size`` label.
        """
        if not contours:
            return []

        # ── Step 1: collect all cutline-sized contours with centroids ──
        _CUTLINE_MIN_MM = 60.0  # lowered from 100 to catch smaller pieces

        cutline_candidates: list[dict] = []
        small_contours: list[dict] = []

        for c in contours:
            w, h, n = c["width"], c["height"], c["count"]
            maxdim = max(w, h)
            if maxdim > _CUTLINE_MIN_MM and n > 5:
                xs = [v[0] for v in c["vertices"]]
                ys = [v[1] for v in c["vertices"]]
                c["cx"] = (min(xs) + max(xs)) / 2
                c["cy"] = (min(ys) + max(ys)) / 2
                cutline_candidates.append(c)
            else:
                small_contours.append(c)

        # ── Step 2: extract piece names ──
        _META_LIKE = {
            "KUMAS", "ASTAR", "TELA", "CIZIM", "ÇİZİM", "SEZON",
        }

        def _is_piece_name(name: str) -> bool:
            if name in _META_LIKE:
                return False
            if name.replace(".", "").isdigit():
                return False
            if "ADET" in name or "KESILSIN" in name or "TELALI" == name:
                return False
            if len(name) < 2:
                return False
            return True

        seen_pieces: dict[str, int] = {}
        piece_order: list[str] = []
        for off, name in names:
            if _is_piece_name(name) and name not in seen_pieces:
                seen_pieces[name] = off
                piece_order.append(name)

        # ── Step 3: group cutlines by piece name ──
        # Each piece name appears multiple times in the file (once per size).
        # Collect all name occurrences, then for each occurrence find the
        # closest cutline contour → that's this piece at this size.
        all_name_occurrences: list[tuple[str, int]] = [
            (name, off) for off, name in names if _is_piece_name(name)
        ]

        # For each piece name, collect all cutline contours assigned to it
        named_cutlines: dict[str, list[dict]] = {}
        for pname in piece_order:
            # Get ALL occurrences of this name
            occurrences = [off for name, off in all_name_occurrences if name == pname]
            cutlines_for_piece: list[dict] = []
            for occ_off in occurrences:
                # Find the closest cutline contour AFTER this name occurrence
                best_c = None
                best_dist = float("inf")
                for c in cutline_candidates:
                    if c["offset"] > occ_off:
                        dist = c["offset"] - occ_off
                        if dist < best_dist:
                            best_dist = dist
                            best_c = c
                if best_c is not None and best_dist < 500_000:  # within 500KB
                    cutlines_for_piece.append(best_c)
            named_cutlines[pname] = cutlines_for_piece

        # Build named_clusters from deduplicated cutlines per piece
        named_clusters: list[tuple[str, list[dict]]] = []
        for pname in piece_order:
            members = named_cutlines.get(pname, [])
            if members:
                named_clusters.append((pname, members))

        # ── Step 5: deduplicate within clusters ──
        # Within a cluster, contours with identical bbox are duplicates
        # (cutline + drawing layer of same size). Keep the one with most vertices.
        # Contours with different bbox = different sizes.
        def _bbox_key(c: dict) -> tuple:
            # Round to nearest 20mm to merge cutline + drawing layer of same size.
            # Different graded sizes typically differ by >=20mm in at least one dimension.
            return (round(c["width"] / 20) * 20, round(c["height"] / 20) * 20)

        pieces: list[Piece] = []
        piece_id = 0

        for pname, members in named_clusters:
            # Group by bbox → deduplicate
            bbox_groups: dict[tuple, list[dict]] = {}
            for c in members:
                key = _bbox_key(c)
                bbox_groups.setdefault(key, []).append(c)

            # For each unique bbox (= unique size), pick best contour
            size_contours: list[dict] = []
            for key, group in bbox_groups.items():
                best = max(group, key=lambda c: c["count"])
                size_contours.append(best)

            # Sort by area (width * height) — smallest to largest
            size_contours.sort(key=lambda c: c["width"] * c["height"])

            # ── Step 6: build a Piece for each size ──
            num_sizes = len(size_contours)
            for size_idx, cutline_c in enumerate(size_contours):
                cutline_pts = self._tessellate_bezier_contour(cutline_c["vertices"])
                if len(cutline_pts) < 3:
                    continue

                # Find nearby small contours for this specific size
                # (within cutline bbox + small margin)
                cx, cy = cutline_c["cx"], cutline_c["cy"]
                half_w = cutline_c["width"] / 2 + 50
                half_h = cutline_c["height"] / 2 + 50
                nearby = [
                    sc for sc in small_contours
                    if (abs(sc.get("_cx", self._contour_cx(sc)) - cx) < half_w
                        and abs(sc.get("_cy", self._contour_cy(sc)) - cy) < half_h)
                ]

                grainline_c = None
                drill_pts: list[tuple[float, float]] = []
                notch_pts: list[tuple[float, float, float]] = []

                for sc in nearby:
                    w, h, n = sc["width"], sc["height"], sc["count"]
                    maxdim = max(w, h)
                    if n == 2 and maxdim > 50 and min(w, h) < 5:
                        grainline_c = sc
                    elif n == 1 and maxdim < 0.1:
                        vx, vy, _ = sc["vertices"][0]
                        drill_pts.append((vx, vy))
                    elif n == 2 and maxdim < 15:
                        v0, v1 = sc["vertices"][0], sc["vertices"][1]
                        mx = (v0[0] + v1[0]) / 2
                        my = (v0[1] + v1[1]) / 2
                        angle = math.degrees(math.atan2(v1[1] - v0[1], v1[0] - v0[0]))
                        notch_pts.append((mx, my, angle))

                # Build grainline
                grainlines: list[Grainline] = []
                if grainline_c is not None:
                    v0 = grainline_c["vertices"][0]
                    v1 = grainline_c["vertices"][1]
                    gx = (v0[0] + v1[0]) / 2
                    gy = (v0[1] + v1[1]) / 2
                    g_angle = math.degrees(math.atan2(v1[1] - v0[1], v1[0] - v0[0]))
                    g_length = math.hypot(v1[0] - v0[0], v1[1] - v0[1])
                    grainlines.append(Grainline(x=gx, y=gy, angle=g_angle, length=g_length))

                # Build drill points
                drill_points = [DrillPoint(x=dx, y=dy) for dx, dy in drill_pts]

                # Build notches — deduplicate pairs
                notches: list[Notch] = []
                used_notch: set[int] = set()
                for i, (nx, ny, na) in enumerate(notch_pts):
                    if i in used_notch:
                        continue
                    for j in range(i + 1, len(notch_pts)):
                        if j in used_notch:
                            continue
                        ox, oy, _ = notch_pts[j]
                        if math.hypot(nx - ox, ny - oy) < 1.0:
                            used_notch.add(j)
                            break
                    notches.append(Notch(x=nx, y=ny, label="", edge_angle=na))

                # Size label
                if num_sizes == 1:
                    display_name = pname
                    size_label = ""
                else:
                    size_label = str(size_idx + 1)
                    display_name = f"{pname} [{size_label}]"

                piece = self._build_piece(piece_id, display_name, cutline_pts, piece_id)
                piece.size = size_label
                piece.grainlines = grainlines
                piece.drill_points = drill_points
                piece.notches = notches
                pieces.append(piece)
                piece_id += 1

        return pieces

    @staticmethod
    def _contour_cx(c: dict) -> float:
        """Compute and cache centroid X for a contour."""
        if "_cx" not in c:
            xs = [v[0] for v in c["vertices"]]
            c["_cx"] = (min(xs) + max(xs)) / 2
        return c["_cx"]

    @staticmethod
    def _contour_cy(c: dict) -> float:
        """Compute and cache centroid Y for a contour."""
        if "_cy" not in c:
            ys = [v[1] for v in c["vertices"]]
            c["_cy"] = (min(ys) + max(ys)) / 2
        return c["_cy"]

    def _cluster_by_bbox(
        self, contours: list[dict], tolerance: float = 0.1
    ) -> list[list[dict]]:
        """Group contours with similar bounding boxes (within tolerance ratio)."""
        if not contours:
            return []

        clusters: list[list[dict]] = []
        used = set()

        for i, ci in enumerate(contours):
            if i in used:
                continue
            cluster = [ci]
            used.add(i)
            for j, cj in enumerate(contours):
                if j in used:
                    continue
                # Check if bboxes are similar
                wi, hi = ci["width"], ci["height"]
                wj, hj = cj["width"], cj["height"]
                if wi > 0 and hi > 0:
                    w_ratio = abs(wi - wj) / max(wi, wj)
                    h_ratio = abs(hi - hj) / max(hi, hj)
                    if w_ratio < tolerance and h_ratio < tolerance:
                        cluster.append(cj)
                        used.add(j)
            clusters.append(cluster)

        return clusters

    def _tessellate_bezier_contour(
        self, vertices: list[tuple[float, float, int]]
    ) -> list[tuple[float, float]]:
        """Convert Bezier anchor + control-point sequence into a polyline.

        Type bytes:
          0x00 = on-curve anchor
          0xF0 = connecting anchor (treated same as 0x00)
          0x0F = bezier control point

        Sequence: ANCHOR [CP1 CP2 ANCHOR] ... forming cubic Bezier segments.
        Straight segments have no CPs between anchors.
        """
        points: list[tuple[float, float]] = []

        # Separate into anchors and control points preserving order
        # Build segments: each segment is anchor → (optional CP1, CP2) → anchor
        n = len(vertices)
        i = 0

        while i < n:
            x, y, typ = vertices[i]

            if typ != 0x0F:
                # This is an anchor
                if not points or (points[-1][0] != x or points[-1][1] != y):
                    points.append((x, y))
                i += 1

                # Check for following control points
                if i + 2 < n and vertices[i][2] == 0x0F and vertices[i + 1][2] == 0x0F:
                    # Cubic Bezier: current anchor → CP1 → CP2 → next anchor
                    cp1x, cp1y, _ = vertices[i]
                    cp2x, cp2y, _ = vertices[i + 1]
                    i += 2
                    # Next should be an anchor
                    if i < n:
                        ax, ay, _ = vertices[i]
                        # Tessellate cubic Bezier
                        p0 = (x, y)
                        p1 = (cp1x, cp1y)
                        p2 = (cp2x, cp2y)
                        p3 = (ax, ay)
                        for t_idx in range(1, _BEZIER_STEPS + 1):
                            t = t_idx / _BEZIER_STEPS
                            bx, by = _cubic_bezier(p0, p1, p2, p3, t)
                            points.append((bx, by))
                        # Don't increment i — the anchor will be picked up next iteration
            else:
                # Orphan control point — skip
                i += 1

        return points

    # -----------------------------------------------------------------------
    # GEM format (legacy)
    # -----------------------------------------------------------------------

    def _parse_gem(self, zf: zipfile.ZipFile, gem_name: str) -> list[Piece]:
        """Parse legacy GEM format (inner .gem binary with float64 vertices)."""
        data = zf.read(gem_name)
        if len(data) < _GEM_HEADER_SIZE:
            return []

        # 1) Find piece header records
        records = self._find_gem_piece_records(data)
        if not records:
            return []

        # 2) For each piece, extract vertices from its data region
        pieces: list[Piece] = []
        piece_id = 0

        for i, rec in enumerate(records):
            # Data region: from end of this header to start of next piece record
            data_start = rec["data_start"]
            data_end = records[i + 1]["record_offset"] if i + 1 < len(records) else len(data)

            raw_points = self._extract_gem_vertices(data, data_start, data_end)
            if len(raw_points) < 3:
                continue

            # Remove outliers (GEM data can be noisy)
            clean_points = self._remove_outliers(raw_points)
            if len(clean_points) < 3:
                continue

            # Check bbox
            xs = [p[0] for p in clean_points]
            ys = [p[1] for p in clean_points]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w < _MIN_PIECE_BBOX_MM or h < _MIN_PIECE_BBOX_MM:
                continue

            color_hex = "#{:02X}{:02X}{:02X}".format(rec["r"], rec["g"], rec["b"])
            piece = self._build_piece(piece_id, rec["name"], clean_points, piece_id)
            piece.color = color_hex
            pieces.append(piece)
            piece_id += 1

        return pieces

    def _find_gem_piece_records(self, data: bytes) -> list[dict]:
        """Find piece header records in inner GEM binary.

        Record format:
          [u32 pieceId][u32 orderNum][u32 nameLen][name + null][R G B pad][fixed header...]

        nameLen includes the null terminator.
        """
        records: list[dict] = []
        end = len(data) - 30

        offset = _GEM_HEADER_SIZE  # skip global header
        while offset < end:
            # Read candidate nameLen
            name_len = struct.unpack_from("<I", data, offset + 8)[0]
            if not (2 <= name_len <= 20):
                offset += 1
                continue

            # Read name bytes (excluding null terminator)
            name_start = offset + 12
            name_end = name_start + name_len - 1
            if name_end + 1 > len(data):
                offset += 1
                continue

            # Check null terminator
            if data[name_end] != 0:
                offset += 1
                continue

            name_bytes = data[name_start:name_end]
            try:
                name = name_bytes.decode("ascii")
            except (UnicodeDecodeError, ValueError):
                offset += 1
                continue

            # Validate: uppercase letters and digits only
            if not (name and all(c.isupper() or c.isdigit() for c in name) and len(name) >= 2):
                offset += 1
                continue

            # Read piece ID and order
            piece_id_val = struct.unpack_from("<I", data, offset)[0]
            order_val = struct.unpack_from("<I", data, offset + 4)[0]

            if piece_id_val > 200000 or order_val > 5000:
                offset += 1
                continue

            # Color bytes follow the name + null
            color_offset = name_start + name_len
            if color_offset + 4 > len(data):
                offset += 1
                continue

            r, g, b, _pad = struct.unpack_from("BBBB", data, color_offset)

            records.append({
                "record_offset": offset,
                "piece_id": piece_id_val,
                "order": order_val,
                "name": name,
                "r": r,
                "g": g,
                "b": b,
                "data_start": color_offset + 4,
            })

            # Jump past this record
            offset = color_offset + 4
            continue

        return records

    def _extract_gem_vertices(
        self, data: bytes, start: int, end: int
    ) -> list[tuple[float, float]]:
        """Extract float64 vertex pairs from a GEM piece data region.

        Strategy 1: Look for structural [02 00] markers.
          Format: [02 00][u32 vertexId][f64 x][f64 y]

        Strategy 2: Brute-force scan for plausible f64 pairs.
        """
        points = self._extract_gem_vertices_markers(data, start, end)
        if len(points) >= 3:
            return points

        # Fallback: brute-force f64 scan
        return self._extract_gem_vertices_bruteforce(data, start, end)

    def _extract_gem_vertices_markers(
        self, data: bytes, start: int, end: int
    ) -> list[tuple[float, float]]:
        """Extract vertices using [02 00][u32 vid][f64 x][f64 y] markers.

        Only keeps entries where vid > 0 (vid=0 entries are set separators
        or grading data markers, not actual contour vertices).
        """
        marker = b"\x02\x00"
        points: list[tuple[float, float]] = []
        off = start

        while off < end - 22:
            if data[off:off + 2] != marker:
                off += 1
                continue

            vid = struct.unpack_from("<I", data, off + 2)[0]
            x = struct.unpack_from("<d", data, off + 6)[0]
            y = struct.unpack_from("<d", data, off + 14)[0]

            # vid > 0 filters out set separators / grading markers
            if 0 < vid < 10000 and -50000 < x < 50000 and -50000 < y < 50000:
                points.append((x, y))

            off += 22  # skip past this record

        return points

    def _extract_gem_vertices_bruteforce(
        self, data: bytes, start: int, end: int
    ) -> list[tuple[float, float]]:
        """Brute-force scan: try every 8-byte-aligned offset for f64 pairs."""
        points: list[tuple[float, float]] = []

        off = start
        while off + 16 <= end:
            x = struct.unpack_from("<d", data, off)[0]
            y = struct.unpack_from("<d", data, off + 8)[0]

            # Filter: reasonable coordinate range and not NaN/Inf
            if (
                math.isfinite(x) and math.isfinite(y)
                and 100.0 < x < 15000.0
                and 100.0 < y < 15000.0
            ):
                points.append((x, y))
                off += 16
            else:
                off += 8  # try next alignment

        return points

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    def _build_piece(
        self,
        id: int,
        name: str,
        points: list[tuple[float, float]],
        color_idx: int,
    ) -> Piece:
        """Build a Piece from raw contour points.  Compute geometry metrics."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Normalize: shift so that min corner is near origin
        normalized = [(x - min_x, y - min_y) for x, y in points]

        piece = Piece(
            id=id,
            name=name,
            cutline=normalized,
            color=COLORS[color_idx % len(COLORS)],
        )

        piece.bounds = (0.0, 0.0, max_x - min_x, max_y - min_y)
        piece.width = max_x - min_x
        piece.height = max_y - min_y
        piece.area = polygon_area(normalized)
        piece.perimeter = polygon_perimeter(normalized)
        piece.centroid = _centroid(normalized)

        return piece

    def _remove_outliers(
        self, points: list[tuple[float, float]], sigma: float = _OUTLIER_SIGMA
    ) -> list[tuple[float, float]]:
        """MAD-based outlier removal for noisy GEM vertex data.

        Removes points whose distance from the median center exceeds
        sigma * MAD (Median Absolute Deviation).
        """
        if len(points) < 5:
            return points

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        med_x = median(xs)
        med_y = median(ys)

        dists = [math.hypot(x - med_x, y - med_y) for x, y in points]
        med_dist = median(dists)
        mad = median([abs(d - med_dist) for d in dists])

        if mad < 1e-6:
            return points

        threshold = med_dist + sigma * mad * 1.4826  # 1.4826 = consistency constant
        return [
            p for p, d in zip(points, dists)
            if d <= threshold
        ]


# ---------------------------------------------------------------------------
# Module-level helper (used by _tessellate_bezier_contour)
# ---------------------------------------------------------------------------

def _cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate cubic Bezier at parameter t.

    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    """
    u = 1.0 - t
    uu = u * u
    tt = t * t
    uuu = uu * u
    ttt = tt * t

    x = uuu * p0[0] + 3.0 * uu * t * p1[0] + 3.0 * u * tt * p2[0] + ttt * p3[0]
    y = uuu * p0[1] + 3.0 * uu * t * p1[1] + 3.0 * u * tt * p2[1] + ttt * p3[1]
    return x, y
