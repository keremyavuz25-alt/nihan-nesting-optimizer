"""Skyline-BLF decoder v2.1 — LRU cache, multi-candidate, unplaced penalty."""
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


class BLFDecoder:
    """Skyline Bottom-Left Fill v2.1.

    v2.0 bug fix: multi-candidate artık TÜM x pozisyonlarını tarıyor.
    v1'den farklar:
    - Arbitrary açı desteği (LRU cache, 512 entry)
    - Multi-candidate skyline (en düşük top 3'ünden min varyans)
    - Unplaced piece penalty (fitness'ta -5%/parça)
    """

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0,
                 resolution: float = 3.0, cache_size: int = 512):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.res = resolution
        self.n = len(pieces)
        self.bin_w = int(bin_width / resolution)
        self._cache_max = cache_size

        # LRU cache — cardinal açılar warm, arbitrary açılar on-demand
        self._cache = {}
        for i, p in enumerate(pieces):
            for rot in [0, 90, 180, 270]:
                key = (i, float(rot))
                self._cache[key] = self._rasterize_fast(p["polygon"], rot)

    def _rasterize_fast(self, poly: Polygon, rotation: float) -> np.ndarray:
        """Hızlı rasterize — matplotlib path ile."""
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

    def _get_bitmap(self, piece_idx: int, rotation: float) -> np.ndarray:
        """Cache'den bitmap al, yoksa rasterize et ve cache'e ekle."""
        cache_key = (piece_idx, round(rotation % 360, 1))

        if cache_key in self._cache:
            return self._cache[cache_key]

        bmp = self._rasterize_fast(
            self.original_pieces[piece_idx]["polygon"], rotation
        )

        if len(self._cache) < self._cache_max:
            self._cache[cache_key] = bmp

        return bmp

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """Skyline-BLF yerleştirme — multi-candidate (K=3)."""
        max_h = int(sum(max(p["width"], p["height"]) for p in self.original_pieces) * 1.5 / self.res)
        canvas = np.zeros((max_h, self.bin_w), dtype=np.uint8)
        skyline = np.zeros(self.bin_w, dtype=np.int32)

        placed = []

        for idx in sequence:
            rot = rotations[idx]
            bmp = self._get_bitmap(idx, rot)
            bh, bw = bmp.shape

            # Ene sığmıyorsa 90° döndür
            if bw > self.bin_w:
                rot = (rot + 90) % 360
                bmp = self._get_bitmap(idx, rot)
                bh, bw = bmp.shape
                if bw > self.bin_w:
                    continue

            # TÜM geçerli pozisyonları tara (v2.0 bug fix: erken break yok)
            all_valid = []

            for x in range(self.bin_w - bw + 1):
                base_y = int(np.max(skyline[x:x + bw]))

                if base_y + bh >= max_h:
                    continue

                region = canvas[base_y:base_y + bh, x:x + bw]
                if region.shape != bmp.shape:
                    continue

                if not np.any(region & bmp):
                    top = base_y + bh
                    all_valid.append((x, base_y, top))

            if not all_valid:
                # Fallback: en üste koy
                best_y = int(np.max(skyline))
                best_x = 0
                if best_y + bh >= max_h:
                    continue
                all_valid = [(best_x, best_y, best_y + bh)]

            # En düşük top'a göre sırala, ilk 3'ü al
            all_valid.sort(key=lambda c: c[2])
            top3 = all_valid[:3]

            if len(top3) == 1:
                best_x, best_y = top3[0][0], top3[0][1]
            else:
                # Tiebreak: skyline varyansı en düşük olan
                best_candidate = None
                best_score = float("inf")

                for x, y, top in top3:
                    sim_skyline = skyline.copy()
                    for xi in range(x, min(x + bw, self.bin_w)):
                        sim_skyline[xi] = max(sim_skyline[xi], top)
                    variance = float(np.std(sim_skyline))
                    if variance < best_score:
                        best_score = variance
                        best_candidate = (x, y)

                best_x, best_y = best_candidate

            # Yerleştir
            canvas[best_y:best_y + bh, best_x:best_x + bw] |= bmp

            # Skyline güncelle
            for xi in range(best_x, min(best_x + bw, self.bin_w)):
                col = canvas[:, xi]
                nonzero = np.nonzero(col)[0]
                skyline[xi] = (nonzero[-1] + 1) if len(nonzero) > 0 else 0

            placed.append({
                "piece_id": idx,
                "x": best_x * self.res,
                "y": best_y * self.res,
                "rotation": rot,
                "polygon": self._get_placed_polygon(idx, rot,
                                                    best_x * self.res,
                                                    best_y * self.res),
            })

        # Verimlilik
        if not placed:
            return {"placements": [], "used_length": 0, "utilization": 0.0,
                    "total_piece_area": 0, "bin_area": 0, "n_placed": 0}

        used_length = float(np.max(skyline)) * self.res
        total_piece_area = sum(self.original_pieces[p["piece_id"]]["area"] for p in placed)
        bin_area = self.bin_width * used_length if used_length > 0 else 1

        return {
            "placements": placed,
            "used_length": used_length,
            "utilization": (total_piece_area / bin_area * 100) if bin_area > 0 else 0.0,
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
