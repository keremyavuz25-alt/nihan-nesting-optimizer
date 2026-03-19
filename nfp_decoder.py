"""NFP decoder v3 — pyclipper Minkowski sum ile gerçek No-Fit Polygon."""
import numpy as np
import pyclipper
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate


# pyclipper integer scale (mm → micron)
SCALE = 1000


def _to_clipper(coords):
    """Shapely coords → pyclipper integer coords."""
    return [(int(x * SCALE), int(y * SCALE)) for x, y in coords]


def _from_clipper(coords):
    """pyclipper integer coords → float coords."""
    return [(x / SCALE, y / SCALE) for x, y in coords]


def _minkowski_sum(subject_coords, pattern_coords):
    """Minkowski sum via pyclipper — doğru NFP hesabı."""
    result = pyclipper.MinkowskiSum(
        _to_clipper(subject_coords),
        _to_clipper(pattern_coords),
        True  # closed
    )
    if not result:
        return Polygon()

    # En büyük polygon'u al (dış sınır)
    best = max(result, key=lambda p: abs(pyclipper.Area(p)))
    coords = _from_clipper(best)
    if len(coords) < 3:
        return Polygon()

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


class NFPDecoder:
    """No-Fit Polygon decoder v3 — pyclipper tabanlı.

    Gerçek Minkowski sum ile NFP hesabı.
    Konkav parçalar için doğru sonuç verir.
    """

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.n = len(pieces)

        # Caches
        self._rotated_cache = {}
        self._coords_cache = {}
        self._nfp_cache = {}

        # Cardinal açıları ön hesapla
        for i in range(self.n):
            for angle in [0.0, 90.0, 180.0, 270.0]:
                self._get_rotated(i, angle)

    def _get_rotated(self, idx: int, rotation: float) -> Polygon:
        """Döndürülmüş polygon, origin-normalized, cached."""
        qr = round(rotation % 360, 0)
        key = (idx, qr)
        if key in self._rotated_cache:
            return self._rotated_cache[key]

        poly = self.original_pieces[idx]["polygon"]
        if qr != 0:
            poly = rotate(poly, qr, origin="centroid", use_radians=False)
            b = poly.bounds
            poly = translate(poly, -b[0], -b[1])

        self._rotated_cache[key] = poly
        # Coords cache — dış sınır koordinatları (kapalı ring, son nokta hariç)
        coords = list(poly.exterior.coords)[:-1]
        self._coords_cache[key] = coords
        return poly

    def _get_nfp(self, fixed_idx: int, fixed_rot: float,
                 moving_idx: int, moving_rot: float) -> Polygon:
        """İki parça arası NFP — cached."""
        qf = round(fixed_rot % 360, 0)
        qm = round(moving_rot % 360, 0)
        key = (fixed_idx, qf, moving_idx, qm)

        if key in self._nfp_cache:
            return self._nfp_cache[key]

        fixed_coords = self._coords_cache[(fixed_idx, qf)]
        moving_coords = self._coords_cache[(moving_idx, qm)]

        # Moving'i aynala (Minkowski difference = sum with mirrored)
        mirrored = [(-x, -y) for x, y in moving_coords]

        nfp = _minkowski_sum(fixed_coords, mirrored)
        self._nfp_cache[key] = nfp
        return nfp

    def _get_ifr(self, piece: Polygon) -> Polygon:
        """Inner-Fit Rectangle — parçanın bin içinde hareket edebileceği alan."""
        b = piece.bounds
        pw = b[2] - b[0]
        ph = b[3] - b[1]

        if pw > self.bin_width:
            return Polygon()

        # Referans noktası (0,0) nereye gidebilir
        return Polygon([
            (0, 0),
            (self.bin_width - pw, 0),
            (self.bin_width - pw, 50000),  # yeterince uzun
            (0, 50000),
        ])

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """NFP-based BL placement."""
        placed = []
        placed_meta = []  # (idx, rot, x, y)

        for idx in sequence:
            rot = rotations[idx]
            piece = self._get_rotated(idx, rot)
            pw = piece.bounds[2] - piece.bounds[0]

            # Ene sığmıyorsa döndür
            if pw > self.bin_width:
                rot = (rot + 90) % 360
                piece = self._get_rotated(idx, rot)
                pw = piece.bounds[2] - piece.bounds[0]
                if pw > self.bin_width:
                    continue

            ifr = self._get_ifr(piece)
            if ifr.is_empty:
                continue

            if not placed_meta:
                # İlk parça — (0,0)
                pos_x, pos_y = 0.0, 0.0
            else:
                # NFP'leri hesapla ve feasible bölge bul
                nfp_polys = []
                for pl_idx, pl_rot, pl_x, pl_y in placed_meta:
                    nfp = self._get_nfp(pl_idx, pl_rot, idx, rot)
                    if not nfp.is_empty:
                        nfp_translated = translate(nfp, pl_x, pl_y)
                        nfp_polys.append(nfp_translated)

                if nfp_polys:
                    try:
                        nfp_union = pyclipper.Pyclipper()
                        for np_poly in nfp_polys:
                            coords = list(np_poly.exterior.coords)[:-1]
                            if len(coords) >= 3:
                                nfp_union.AddPath(
                                    _to_clipper(coords),
                                    pyclipper.PT_CLIP, True
                                )

                        ifr_coords = list(ifr.exterior.coords)[:-1]
                        nfp_union.AddPath(
                            _to_clipper(ifr_coords),
                            pyclipper.PT_SUBJECT, True
                        )

                        feasible_paths = nfp_union.Execute(
                            pyclipper.CT_DIFFERENCE,
                            pyclipper.PFT_NONZERO,
                            pyclipper.PFT_NONZERO,
                        )

                        if feasible_paths:
                            pos_x, pos_y = self._bottom_left_from_paths(feasible_paths)
                        else:
                            # Fallback
                            top_y = max(pl_y + self._get_rotated(pl_idx, pl_rot).bounds[3]
                                       for pl_idx, pl_rot, pl_x, pl_y in placed_meta)
                            pos_x, pos_y = 0.0, top_y + 1.0
                    except pyclipper.ClipperException:
                        top_y = max(pl_y + self._get_rotated(pl_idx, pl_rot).bounds[3]
                                   for pl_idx, pl_rot, pl_x, pl_y in placed_meta)
                        pos_x, pos_y = 0.0, top_y + 1.0
                else:
                    pos_x, pos_y = 0.0, 0.0

            placed_poly = translate(piece, pos_x, pos_y)
            placed_meta.append((idx, rot, pos_x, pos_y))
            placed.append({
                "piece_id": idx,
                "x": pos_x,
                "y": pos_y,
                "rotation": rot,
                "polygon": placed_poly,
            })

        # Verimlilik
        if not placed:
            return {"placements": [], "used_length": 0, "utilization": 0.0,
                    "total_piece_area": 0, "bin_area": 0, "n_placed": 0}

        used_length = max(p["polygon"].bounds[3] for p in placed)
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

    @staticmethod
    def _bottom_left_from_paths(paths) -> tuple:
        """pyclipper path'lerinden en alt-sol noktayı bul."""
        best_y = float("inf")
        best_x = float("inf")

        for path in paths:
            coords = _from_clipper(path)
            for x, y in coords:
                if y < best_y or (abs(y - best_y) < 0.5 and x < best_x):
                    best_y = y
                    best_x = x

        return (best_x, best_y) if best_y != float("inf") else (0.0, 0.0)

    def fitness(self, sequence: list[int], rotations: list[float]) -> float:
        """Fitness = utilization - unplaced penalty."""
        result = self.decode(sequence, rotations)
        base = result["utilization"]
        unplaced = self.n - result["n_placed"]
        penalty = unplaced * 5.0
        return max(0.0, base - penalty)
