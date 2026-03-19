"""Skyline-BLF decoder v3.0 — optimized CPU decoder."""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from matplotlib.path import Path as MplPath


class BLFDecoder:
    """Skyline Bottom-Left Fill v3.0 — optimized.

    Same placement decisions as v2.1, with:
    - Vectorized sliding window max for skyline queries
    - Batch collision detection per base_y level
    - Incremental skyline update via per-column bitmap heights
    - Angle quantization + large cache
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
        self._angle_step = angle_step

        self._max_h = int(sum(max(p["width"], p["height"])
                              for p in pieces) * 1.5 / resolution)

        self._cache = {}
        for i, p in enumerate(pieces):
            for rot in [0, 90, 180, 270]:
                key = (i, float(rot))
                bmp = self._rasterize(p["polygon"], rot)
                self._cache[key] = (bmp, self._col_heights(bmp))

    def _rasterize(self, poly: Polygon, rotation: float) -> np.ndarray:
        """Rasterize using matplotlib Path."""
        if rotation != 0:
            poly = rotate(poly, rotation, origin="centroid", use_radians=False)
            b = poly.bounds
            poly = translate(poly, -b[0], -b[1])

        b = poly.bounds
        res = self.res
        w = int(np.ceil((b[2] - b[0]) / res)) + 2
        h = int(np.ceil((b[3] - b[1]) / res)) + 2

        if w <= 0 or h <= 0:
            return np.zeros((1, 1), dtype=np.uint8)

        coords = np.array(poly.exterior.coords)
        path = MplPath(coords)

        xs = np.arange(w) * res + b[0]
        ys = np.arange(h) * res + b[1]
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        mask = path.contains_points(points).reshape(h, w)
        return mask.astype(np.uint8)

    @staticmethod
    def _col_heights(bmp: np.ndarray) -> np.ndarray:
        """Per-column: highest active row + 1 (0 if empty)."""
        bh, bw = bmp.shape
        row_indices = np.arange(bh).reshape(-1, 1)
        masked = np.where(bmp > 0, row_indices, -1)
        heights = np.max(masked, axis=0) + 1
        heights = np.clip(heights, 0, bh)
        return heights.astype(np.int32)

    def _get_bitmap(self, piece_idx: int, rotation: float):
        """Get (bitmap, col_heights) from cache or compute."""
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
        """Skyline-BLF placement — same decisions as v2.1, faster."""
        pieces = self.original_pieces
        res = self.res
        bin_w = self.bin_w
        max_h = self._max_h

        canvas = np.zeros((max_h, bin_w), dtype=np.uint8)
        skyline = np.zeros(bin_w, dtype=np.int32)

        placed = []

        for idx in sequence:
            rot = rotations[idx]
            bmp, col_h = self._get_bitmap(idx, rot)
            bh, bw = bmp.shape

            if bw > bin_w:
                rot = (rot + 90) % 360
                bmp, col_h = self._get_bitmap(idx, rot)
                bh, bw = bmp.shape
                if bw > bin_w:
                    continue

            x_range = bin_w - bw + 1
            if x_range <= 0:
                continue

            # Vectorized sliding window max
            if bw == 1:
                base_ys = skyline[:x_range].copy()
            else:
                sky_windows = as_strided(
                    skyline,
                    shape=(x_range, bw),
                    strides=(skyline.strides[0], skyline.strides[0]))
                base_ys = np.max(sky_windows, axis=1)

            # --- BATCH COLLISION DETECTION ---
            # Group x positions by base_y and check collisions in bulk
            all_valid = []

            # Get unique base_y values and their x positions
            unique_bys = np.unique(base_ys)

            for by_val in unique_bys:
                by = int(by_val)
                if by + bh >= max_h:
                    continue

                # All x positions at this base_y level
                xs_at_level = np.nonzero(base_ys == by_val)[0]

                if len(xs_at_level) == 0:
                    continue

                # Extract the canvas strip at this y level
                strip = canvas[by:by + bh, :]  # shape (bh, bin_w)

                # For each x in xs_at_level, check if strip[:, x:x+bw] & bmp has any overlap
                # Vectorize: create sliding windows over the strip
                if len(xs_at_level) <= 3:
                    # Few positions: check individually (faster than windowing overhead)
                    for x in xs_at_level:
                        x = int(x)
                        region = strip[:, x:x + bw]
                        if not np.any(region & bmp):
                            all_valid.append((x, by, by + bh))
                else:
                    # Many positions: use vectorized sliding window collision
                    # Create windows of the strip: shape (x_range, bh, bw)
                    strip_windows = as_strided(
                        strip,
                        shape=(x_range, bh, bw),
                        strides=(strip.strides[1], strip.strides[0], strip.strides[1]))

                    # Only check positions in xs_at_level
                    batch = strip_windows[xs_at_level]  # shape (n, bh, bw)

                    # Batch collision: AND with bmp, then check any per sample
                    collisions = batch & bmp[np.newaxis, :, :]  # broadcast
                    has_collision = np.any(collisions.reshape(len(xs_at_level), -1), axis=1)

                    # Collect valid positions
                    free_mask = ~has_collision
                    free_xs = xs_at_level[free_mask]
                    top = by + bh
                    for x in free_xs:
                        all_valid.append((int(x), by, top))

            if not all_valid:
                best_y = int(np.max(skyline))
                best_x = 0
                if best_y + bh >= max_h:
                    continue
                all_valid = [(best_x, best_y, best_y + bh)]

            # Top 3 + variance tiebreak
            all_valid.sort(key=lambda c: c[2])
            top3 = all_valid[:3]

            if len(top3) == 1:
                best_x, best_y = top3[0][0], top3[0][1]
            else:
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

            # Place piece
            canvas[best_y:best_y + bh, best_x:best_x + bw] |= bmp

            # Incremental skyline update
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
