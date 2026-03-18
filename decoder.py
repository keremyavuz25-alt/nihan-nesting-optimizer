"""Skyline-BLF decoder — O(n) placement, numpy raster collision."""
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


class BLFDecoder:
    """Skyline Bottom-Left Fill — skyline profile ile hızlı yerleştirme.

    Her adımda skyline (üst profil) takip edilir.
    Yeni parça skyline'ın en alçak noktasına yerleştirilir.
    Collision check: numpy bitmap AND operasyonu.
    """

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0,
                 resolution: float = 3.0):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.res = resolution
        self.n = len(pieces)
        self.bin_w = int(bin_width / resolution)

        # Bitmap cache (4 standart rotasyon)
        self._cache = {}
        for i, p in enumerate(pieces):
            for rot in [0, 90, 180, 270]:
                bmp = self._rasterize_fast(p["polygon"], rot)
                self._cache[(i, rot)] = bmp

    def _rasterize_fast(self, poly: Polygon, rotation: float) -> np.ndarray:
        """Hızlı rasterize — matplotlib path kullanarak."""
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

        # Grid noktaları oluştur
        xs = np.arange(w) * self.res + b[0]
        ys = np.arange(h) * self.res + b[1]
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        mask = path.contains_points(points).reshape(h, w)
        return mask.astype(np.uint8)

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """Skyline-BLF yerleştirme."""
        max_h = int(sum(max(p["width"], p["height"]) for p in self.original_pieces) * 1.5 / self.res)
        canvas = np.zeros((max_h, self.bin_w), dtype=np.uint8)

        # Skyline: her x kolonu için en üst dolu y
        skyline = np.zeros(self.bin_w, dtype=np.int32)

        placed = []

        for idx in sequence:
            rot = rotations[idx]
            rot_key = rot if rot in (0, 90, 180, 270) else None

            if rot_key is not None and (idx, int(rot_key)) in self._cache:
                bmp = self._cache[(idx, int(rot_key))]
            else:
                bmp = self._rasterize_fast(self.original_pieces[idx]["polygon"], rot)

            bh, bw = bmp.shape

            if bw > self.bin_w:
                alt_rot = (rot + 90) % 360
                alt_key = alt_rot if alt_rot in (0, 90, 180, 270) else None
                if alt_key is not None and (idx, int(alt_key)) in self._cache:
                    bmp = self._cache[(idx, int(alt_key))]
                else:
                    bmp = self._rasterize_fast(self.original_pieces[idx]["polygon"], alt_rot)
                bh, bw = bmp.shape
                rot = alt_rot
                if bw > self.bin_w:
                    continue

            # Skyline-BLF: her x pozisyonunda minimum başlangıç y'yi bul
            best_x, best_y = 0, max_h
            best_top = max_h

            for x in range(self.bin_w - bw + 1):
                # Bu x'te parça tabanının oturması gereken y
                base_y = int(np.max(skyline[x:x + bw]))

                if base_y + bh >= max_h:
                    continue

                # Bitmap collision check
                region = canvas[base_y:base_y + bh, x:x + bw]
                if region.shape != bmp.shape:
                    continue

                overlap = np.any(region & bmp)
                if not overlap:
                    top = base_y + bh
                    if top < best_top:
                        best_top = top
                        best_x = x
                        best_y = base_y

            if best_top >= max_h:
                # Fallback
                best_y = int(np.max(skyline))
                best_x = 0
                if best_y + bh >= max_h:
                    continue

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
                    "total_piece_area": 0, "bin_area": 0}

        used_length = float(np.max(skyline)) * self.res
        total_piece_area = sum(self.original_pieces[p["piece_id"]]["area"] for p in placed)
        bin_area = self.bin_width * used_length if used_length > 0 else 1

        return {
            "placements": placed,
            "used_length": used_length,
            "utilization": (total_piece_area / bin_area * 100) if bin_area > 0 else 0.0,
            "total_piece_area": total_piece_area,
            "bin_area": bin_area,
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
        return self.decode(sequence, rotations)["utilization"]
