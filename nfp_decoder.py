"""NFP-based decoder v2 — optimized No-Fit Polygon placement.

Optimizations over v1:
1. NFP cache with rotation quantization (1-degree bins) for high hit rate
2. Vectorized Minkowski sum via numpy broadcasting (no Python loops)
3. GEOS convex_hull (C-level, faster than scipy for <5K points)
4. Batch NFP subtraction: unary_union + single difference
5. Precomputed concavity flags (skips buffer for convex-only pairs)
6. Numpy coord caching (avoids Shapely->numpy conversion)
7. Boundary vertex + midpoint sampling for bottom-left search
8. Feasible region simplification for complex geometries
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.affinity import rotate, translate
from shapely.ops import unary_union


class NFPDecoder:
    """No-Fit Polygon tabanli nesting decoder v2 — optimized."""

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0,
                 nfp_angles: int = 8, simplify_tol: float = 0.5):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.n = len(pieces)
        self.nfp_angles = nfp_angles
        self.simplify_tol = simplify_tol

        # Caches
        self._nfp_cache = {}
        self._rotated_cache = {}
        self._coords_cache = {}

        # Precompute concavity flags (area-based, avoids expensive equals())
        self._is_convex = {}
        for i, p in enumerate(pieces):
            poly = p["polygon"]
            self._is_convex[i] = abs(poly.convex_hull.area - poly.area) < 1.0

        # Precompute cardinal rotations + coords
        for i in range(self.n):
            for angle in [0.0, 90.0, 180.0, 270.0]:
                self._get_rotated(i, angle)

    @staticmethod
    def _qrot(rotation: float) -> float:
        """Quantize rotation to 1-degree bins."""
        return round(rotation % 360, 0)

    def _get_rotated(self, piece_idx: int, rotation: float) -> Polygon:
        """Rotated polygon, origin-normalized, cached."""
        qr = self._qrot(rotation)
        key = (piece_idx, qr)
        if key in self._rotated_cache:
            return self._rotated_cache[key]

        poly = self.original_pieces[piece_idx]["polygon"]
        if qr != 0:
            poly = rotate(poly, qr, origin="centroid", use_radians=False)
            b = poly.bounds
            poly = translate(poly, -b[0], -b[1])

        self._rotated_cache[key] = poly
        self._coords_cache[key] = np.asarray(poly.exterior.coords)
        return poly

    def _get_coords(self, piece_idx: int, rotation: float) -> np.ndarray:
        """Numpy coords for rotated piece (cached)."""
        key = (piece_idx, self._qrot(rotation))
        if key not in self._coords_cache:
            self._get_rotated(piece_idx, rotation)
        return self._coords_cache[key]

    def _compute_nfp(self, fixed_idx: int, fixed_rot: float,
                     moving_idx: int, moving_rot: float) -> Polygon:
        """Vectorized Minkowski sum NFP via numpy + GEOS convex_hull."""
        fc = self._get_coords(fixed_idx, fixed_rot)
        mc = self._get_coords(moving_idx, moving_rot)

        # Vectorized: all_points[i,j] = fc[i] + (-mc[j])
        all_points = (fc[:, np.newaxis, :] - mc[np.newaxis, :, :]).reshape(-1, 2)

        if len(all_points) < 3:
            return Polygon()

        # GEOS convex_hull (C-level, fast for <5K points)
        nfp = MultiPoint(all_points).convex_hull

        if nfp.is_empty or not isinstance(nfp, Polygon):
            return Polygon()

        # Only buffer if at least one piece is concave
        if not self._is_convex.get(fixed_idx, True) or not self._is_convex.get(moving_idx, True):
            nfp = nfp.buffer(0.5, join_style=2)

        return nfp

    def _get_nfp(self, fixed_idx: int, fixed_rot: float,
                 moving_idx: int, moving_rot: float,
                 offset_x: float, offset_y: float) -> Polygon:
        """Cached NFP, translated to placed position."""
        qf = self._qrot(fixed_rot)
        qm = self._qrot(moving_rot)
        key = (fixed_idx, qf, moving_idx, qm)

        if key not in self._nfp_cache:
            self._nfp_cache[key] = self._compute_nfp(fixed_idx, fixed_rot,
                                                      moving_idx, moving_rot)

        nfp = self._nfp_cache[key]
        if nfp.is_empty:
            return nfp
        if offset_x != 0 or offset_y != 0:
            return translate(nfp, offset_x, offset_y)
        return nfp

    def _get_ifr(self, pw: float) -> Polygon:
        """Inner-Fit Rectangle for piece of width pw."""
        if pw > self.bin_width:
            return Polygon()
        return Polygon([(0, 0), (self.bin_width - pw, 0),
                        (self.bin_width - pw, 10000), (0, 10000)])

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """NFP-based placement — optimized."""
        placed = []
        placed_polys = []
        placed_meta = []  # (piece_idx, rot, offset_x, offset_y)

        for idx in sequence:
            rot = rotations[idx]
            piece = self._get_rotated(idx, rot)
            pw = piece.bounds[2] - piece.bounds[0]

            # Rotate 90 if too wide
            if pw > self.bin_width:
                rot = (rot + 90) % 360
                piece = self._get_rotated(idx, rot)
                pw = piece.bounds[2] - piece.bounds[0]
                if pw > self.bin_width:
                    continue

            ifr = self._get_ifr(pw)
            if ifr.is_empty:
                continue

            if not placed_polys:
                pos_x, pos_y = 0.0, 0.0
            else:
                # Collect NFPs and do batch subtraction
                nfp_list = []
                for pl_idx, pl_rot, pl_ox, pl_oy in placed_meta:
                    nfp = self._get_nfp(pl_idx, pl_rot, idx, rot, pl_ox, pl_oy)
                    if not nfp.is_empty:
                        nfp_list.append(nfp)

                if nfp_list:
                    feasible = ifr.difference(unary_union(nfp_list))
                else:
                    feasible = ifr

                # Simplify if geometry is complex
                if not feasible.is_empty:
                    if isinstance(feasible, MultiPolygon):
                        nc = sum(len(g.exterior.coords) for g in feasible.geoms)
                    elif isinstance(feasible, Polygon):
                        nc = len(feasible.exterior.coords)
                    else:
                        nc = 0
                    if nc > 200:
                        feasible = feasible.simplify(self.simplify_tol,
                                                     preserve_topology=True)

                if feasible.is_empty:
                    top_y = max(pl.bounds[3] for pl in placed_polys)
                    pos_x, pos_y = 0.0, top_y + 1.0
                else:
                    pos_x, pos_y = self._bottom_left(feasible)

            placed_poly = translate(piece, pos_x, pos_y)
            placed_polys.append(placed_poly)
            placed_meta.append((idx, rot, pos_x, pos_y))

            placed.append({
                "piece_id": idx,
                "x": pos_x,
                "y": pos_y,
                "rotation": rot,
                "polygon": placed_poly,
            })

        if not placed:
            return {"placements": [], "used_length": 0, "utilization": 0.0,
                    "total_piece_area": 0, "bin_area": 0, "n_placed": 0}

        used_length = max(p["polygon"].bounds[3] for p in placed)
        total_piece_area = sum(self.original_pieces[p["piece_id"]]["area"]
                               for p in placed)
        bin_area = self.bin_width * used_length if used_length > 0 else 1

        return {
            "placements": placed,
            "used_length": used_length,
            "utilization": (total_piece_area / bin_area * 100) if bin_area > 0 else 0.0,
            "total_piece_area": total_piece_area,
            "bin_area": bin_area,
            "n_placed": len(placed),
        }

    @staticmethod
    def _bottom_left(feasible) -> tuple:
        """Bottom-left point in feasible region (vertex + midpoint sampling)."""
        if isinstance(feasible, MultiPolygon):
            geoms = feasible.geoms
        elif isinstance(feasible, Polygon):
            geoms = [feasible]
        else:
            return (0.0, 0.0)

        best_y = float("inf")
        best_x = float("inf")

        for geom in geoms:
            if geom.is_empty:
                continue
            coords = np.asarray(geom.exterior.coords)
            if len(coords) == 0:
                continue

            min_y = coords[:, 1].min()
            if min_y > best_y + 1.0:
                continue

            # Bottom vertices
            mask = coords[:, 1] <= min_y + 1.0
            cands = coords[mask]
            if len(cands) > 0:
                li = cands[:, 0].argmin()
                cy, cx = cands[li, 1], cands[li, 0]
                if cy < best_y or (abs(cy - best_y) < 1.0 and cx < best_x):
                    best_y = cy
                    best_x = cx

            # Bottom edge midpoints
            for i in range(len(coords) - 1):
                y1, y2 = coords[i, 1], coords[i + 1, 1]
                if min(y1, y2) <= best_y + 2.0:
                    mx = (coords[i, 0] + coords[i + 1, 0]) * 0.5
                    my = (y1 + y2) * 0.5
                    if my < best_y or (abs(my - best_y) < 1.0 and mx < best_x):
                        if geom.contains(Point(mx, my)):
                            best_y = my
                            best_x = mx

        return (best_x, best_y) if best_y != float("inf") else (0.0, 0.0)

    def fitness(self, sequence: list[int], rotations: list[float]) -> float:
        """Fitness = utilization - unplaced penalty."""
        result = self.decode(sequence, rotations)
        base = result["utilization"]
        unplaced = self.n - result["n_placed"]
        penalty = unplaced * 5.0
        return max(0.0, base - penalty)

    def batch_fitness(self, sequences: list[list[int]],
                      rotations: list[list[float]]) -> list[float]:
        """Batch fitness evaluation using thread pool.

        Shapely/GEOS releases the GIL for C-level geometry ops, so threads
        provide real parallelism for the expensive parts (convex_hull, buffer,
        difference, unary_union). Each individual is decoded independently.
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        n_workers = min(len(sequences), os.cpu_count() or 4)

        if n_workers <= 1 or len(sequences) <= 4:
            return [self.fitness(s, r) for s, r in zip(sequences, rotations)]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(self.fitness, s, r)
                       for s, r in zip(sequences, rotations)]
            return [f.result() for f in futures]
