"""Skyline-BLF decoder v3.0 — optimized CPU decoder."""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


class BLFDecoder:
    """Skyline Bottom-Left Fill v3.0 — optimized.

    Optimizations over v2.1:
    - Vectorized sliding window max for skyline queries (eliminates per-x np.max)
    - Per-column bitmap height for precise O(1) incremental skyline update
    - Early exit in x-scan (sorted by base_y, prune when top >= best_top)
    - Angle quantization for higher cache hit rate
    - Full scan preserved for correctness (same results as v2.1)
    """

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0,
                 resolution: float = 3.0, cache_size: int = 4096,
                 angle_step: float = 1.0):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.res = resolution
        self.n = len(pieces)
        self.bin_w = int(bin_width / resolution)
        self._cache_max = cache_size
        self._angle_step = angle_step  # quantize angles to this step

        # Cache: key → (bitmap, col_heights)
        self._cache = {}
        for i, p in enumerate(pieces):
            for rot in [0, 90, 180, 270]:
                key = (i, float(rot))
                bmp = self._rasterize(p["polygon"], rot)
                self._cache[key] = (bmp, self._col_heights(bmp))

    def _rasterize(self, poly: Polygon, rotation: float) -> np.ndarray:
        """Rasterize using matplotlib Path (proven accurate)."""
        if rotation != 0:
            poly = rotate(poly, rotation, origin="centroid", use_radians=False)
            b = poly.bounds
            poly = translate(poly, -b[0], -b[1])

        b = poly.bounds
        w = int(np.ceil((b[2] - b[0]) / self.res)) + 2
        h = int(np.ceil((b[3] - b[1]) / self.res)) + 2

        if w <= 0 or h <= 0:
            return np.zeros((1, 1), dtype=np.uint8)

        from matplotlib.path import Path
        coords = np.array(poly.exterior.coords)
        path = Path(coords)

        xs = np.arange(w) * self.res + b[0]
        ys = np.arange(h) * self.res + b[1]
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        mask = path.contains_points(points).reshape(h, w)
        return mask.astype(np.uint8)

    @staticmethod
    def _col_heights(bmp: np.ndarray) -> np.ndarray:
        """Per-column: highest active row + 1 (0 if empty)."""
        bh, bw = bmp.shape
        # Reverse row indices: row 0 is bottom, row bh-1 is top
        # Find the last nonzero row per column
        heights = np.zeros(bw, dtype=np.int32)
        for r in range(bh - 1, -1, -1):
            active = bmp[r] > 0
            update = active & (heights == 0)
            heights[update] = r + 1
            if np.all(heights > 0):
                break
        return heights

    def _get_bitmap(self, piece_idx: int, rotation: float):
        """Get (bitmap, col_heights) from cache or compute."""
        # Quantize angle for better cache hits
        step = self._angle_step
        q_rot = round(rotation / step) * step
        q_rot = round(q_rot % 360, 1)
        cache_key = (piece_idx, q_rot)

        if cache_key in self._cache:
            return self._cache[cache_key]

        bmp = self._rasterize(
            self.original_pieces[piece_idx]["polygon"], q_rot)
        col_h = self._col_heights(bmp)
        entry = (bmp, col_h)

        if len(self._cache) < self._cache_max:
            self._cache[cache_key] = entry

        return entry

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """Skyline-BLF placement — fully correct, optimized hot loop."""
        pieces = self.original_pieces
        res = self.res
        bin_w = self.bin_w

        max_h = int(sum(max(p["width"], p["height"])
                        for p in pieces) * 1.5 / res)
        canvas = np.zeros((max_h, bin_w), dtype=np.uint8)
        skyline = np.zeros(bin_w, dtype=np.int32)

        placed = []

        for idx in sequence:
            rot = rotations[idx]
            bmp, col_h = self._get_bitmap(idx, rot)
            bh, bw = bmp.shape

            # Doesn't fit width — try +90deg
            if bw > bin_w:
                rot = (rot + 90) % 360
                bmp, col_h = self._get_bitmap(idx, rot)
                bh, bw = bmp.shape
                if bw > bin_w:
                    continue

            x_range = bin_w - bw + 1
            if x_range <= 0:
                continue

            # === OPTIMIZATION 1: Vectorized sliding window max ===
            # Replaces per-x np.max(skyline[x:x+bw]) loop
            if bw == 1:
                base_ys = skyline[:x_range].astype(np.int64)
            else:
                sky_windows = as_strided(
                    skyline,
                    shape=(x_range, bw),
                    strides=(skyline.strides[0], skyline.strides[0]))
                base_ys = np.max(sky_windows, axis=1)

            # Filter overflow positions
            valid_mask = (base_ys + bh) < max_h
            valid_xs = np.nonzero(valid_mask)[0]

            all_valid = []

            if len(valid_xs) > 0:
                # === OPTIMIZATION 2: Sort by base_y for early exit ===
                sorted_order = np.argsort(base_ys[valid_xs])
                sorted_xs = valid_xs[sorted_order]
                sorted_bys = base_ys[sorted_xs]

                best_top = max_h

                # Full scan with early exit
                for i in range(len(sorted_xs)):
                    by = int(sorted_bys[i])
                    top = by + bh

                    # Early exit: sorted ascending, can't find lower top
                    if top > best_top:
                        break

                    x = int(sorted_xs[i])
                    region = canvas[by:by + bh, x:x + bw]
                    if not np.any(region & bmp):
                        all_valid.append((x, by, top))
                        if top < best_top:
                            best_top = top

            if not all_valid:
                # Fallback: place at absolute top
                best_y = int(np.max(skyline))
                best_x = 0
                if best_y + bh >= max_h:
                    continue
                all_valid = [(best_x, best_y, best_y + bh)]

            # Best from top 3 candidates (same logic as v2.1)
            all_valid.sort(key=lambda c: c[2])
            top3 = all_valid[:3]

            if len(top3) == 1:
                best_x, best_y = top3[0][0], top3[0][1]
            else:
                # Tiebreak: min skyline std-dev
                best_candidate = None
                best_score = float("inf")
                for x, y, top in top3:
                    sim_skyline = skyline.copy()
                    sim_skyline[x:min(x + bw, bin_w)] = np.maximum(
                        sim_skyline[x:min(x + bw, bin_w)], top)
                    score = float(np.std(sim_skyline))
                    if score < best_score:
                        best_score = score
                        best_candidate = (x, y)
                best_x, best_y = best_candidate

            # Place piece on canvas
            canvas[best_y:best_y + bh, best_x:best_x + bw] |= bmp

            # === OPTIMIZATION 3: Incremental skyline update ===
            # Use per-column bitmap heights instead of scanning full columns
            # Only update columns that have active pixels (col_h > 0)
            active_cols = col_h > 0
            sl = skyline[best_x:best_x + bw]
            sl[active_cols] = np.maximum(sl[active_cols],
                                         best_y + col_h[active_cols])

            placed.append({
                "piece_id": idx,
                "x": best_x * res,
                "y": best_y * res,
                "rotation": rot,
                "polygon": self._get_placed_polygon(idx, rot,
                                                    best_x * res,
                                                    best_y * res),
            })

        # Results
        if not placed:
            return {"placements": [], "used_length": 0, "utilization": 0.0,
                    "total_piece_area": 0, "bin_area": 0, "n_placed": 0}

        used_length = float(np.max(skyline)) * res
        total_piece_area = sum(
            pieces[p["piece_id"]]["area"] for p in placed)
        bin_area = self.bin_width * used_length if used_length > 0 else 1

        return {
            "placements": placed,
            "used_length": used_length,
            "utilization": (total_piece_area / bin_area * 100)
                           if bin_area > 0 else 0.0,
            "total_piece_area": total_piece_area,
            "bin_area": bin_area,
            "n_placed": len(placed),
        }

    def _get_placed_polygon(self, piece_idx: int, rotation: float,
                            x: float, y: float) -> Polygon:
        poly = self.original_pieces[piece_idx]["polygon"]
        if rotation != 0:
            poly = rotate(poly, rotation, origin="centroid", use_radians=False)
            b = poly.bounds
            poly = translate(poly, -b[0], -b[1])
        return translate(poly, x, y)

    def fitness(self, sequence: list[int], rotations: list[float]) -> float:
        """Fitness = utilization - unplaced penalty."""
        result = self.decode(sequence, rotations)
        base = result["utilization"]
        unplaced = self.n - result["n_placed"]
        penalty = unplaced * 5.0
        return max(0.0, base - penalty)
