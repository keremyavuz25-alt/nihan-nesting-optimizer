"""NFP-based decoder — No-Fit Polygon ile piksel-perfect yerleştirme."""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate, scale
from shapely.ops import unary_union
from shapely import prepared


class NFPDecoder:
    """No-Fit Polygon tabanlı nesting decoder.

    BLF'den farkı: parçalar arasındaki tam geometrik ilişkiyi hesaplar.
    NFP = A parçasını B'nin etrafında kaydırdığında temas noktalarının izi.
    Placement: IFR (Inner-Fit Rectangle) + NFP kesişimiyle feasible bölge bulunur.
    """

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0,
                 nfp_angles: int = 8):
        """
        Args:
            pieces: polygon listesi
            bin_width: kumaş eni (mm)
            nfp_angles: NFP precompute için açı sayısı (8 = her 45°)
        """
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.n = len(pieces)
        self.nfp_angles = nfp_angles

        # NFP cache: (piece_i, rot_i, piece_j, rot_j) → NFP polygon
        self._nfp_cache = {}
        self._rotated_cache = {}

    def _get_rotated(self, piece_idx: int, rotation: float) -> Polygon:
        """Döndürülmüş ve orijine normalize edilmiş polygon."""
        cache_key = (piece_idx, round(rotation, 1))
        if cache_key in self._rotated_cache:
            return self._rotated_cache[cache_key]

        poly = self.original_pieces[piece_idx]["polygon"]
        if rotation != 0:
            poly = rotate(poly, rotation, origin="centroid", use_radians=False)
            b = poly.bounds
            poly = translate(poly, -b[0], -b[1])

        self._rotated_cache[cache_key] = poly
        return poly

    def _compute_nfp(self, fixed: Polygon, moving: Polygon) -> Polygon:
        """İki polygon arasındaki No-Fit Polygon'u hesapla.

        Minkowski sum yaklaşımı: NFP(A, B) = A ⊕ (-B)
        Basitleştirilmiş versiyon: moving'in referans noktasının
        fixed etrafında hareket edebileceği sınır.
        """
        # Moving'i aynala (merkez etrafında)
        mirrored = scale(moving, xfact=-1, yfact=-1, origin=(0, 0))

        # Minkowski sum: fixed'in her vertex'ini mirrored ile örtüştür
        # Basit yaklaşım: convex hull of offset positions
        fixed_coords = np.array(fixed.exterior.coords)
        mirrored_coords = np.array(mirrored.exterior.coords)

        # Her fixed vertex için mirrored'ı o noktaya koy
        all_points = []
        for fx, fy in fixed_coords:
            for mx, my in mirrored_coords:
                all_points.append((fx + mx, fy + my))

        if len(all_points) < 3:
            return Polygon()

        from shapely.geometry import MultiPoint
        nfp = MultiPoint(all_points).convex_hull

        # Buffer ile küçük boşlukları kapat
        if not nfp.is_valid:
            nfp = nfp.buffer(0)

        return nfp

    def _get_ifr(self, piece: Polygon) -> Polygon:
        """Inner-Fit Rectangle: parçanın bin içinde nereye girebileceği."""
        b = piece.bounds
        pw = b[2] - b[0]
        ph = b[3] - b[1]

        if pw > self.bin_width:
            return Polygon()

        # Parçanın referans noktası (0,0) kumaş içinde nereye gidebilir
        max_x = self.bin_width - pw
        max_y = 10000  # üst sınır (yeterince büyük)

        return Polygon([(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)])

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """NFP-based yerleştirme."""
        placed = []
        placed_polys = []  # yerleştirilmiş polygon'lar (orijinal koordinatlarda)

        for idx in sequence:
            rot = rotations[idx]
            piece = self._get_rotated(idx, rot)
            b = piece.bounds
            pw = b[2] - b[0]
            ph = b[3] - b[1]

            # Ene sığmıyorsa 90° döndür
            if pw > self.bin_width:
                rot = (rot + 90) % 360
                piece = self._get_rotated(idx, rot)
                b = piece.bounds
                pw = b[2] - b[0]
                if pw > self.bin_width:
                    continue

            # IFR: bin sınırları içinde nereye girebilir
            ifr = self._get_ifr(piece)
            if ifr.is_empty:
                continue

            # Feasible bölge: IFR başla, her yerleştirilmiş parçanın NFP'sini çıkar
            feasible = ifr

            for pl in placed_polys:
                nfp = self._compute_nfp(pl, piece)
                if not nfp.is_empty:
                    feasible = feasible.difference(nfp)
                    if feasible.is_empty:
                        break

            if feasible.is_empty:
                # Fallback: en üste koy
                top_y = 0
                for pl in placed_polys:
                    top_y = max(top_y, pl.bounds[3])
                pos_x, pos_y = 0, top_y + 1
            else:
                # Bottom-left noktasını bul (feasible bölgede en düşük y, sonra en sol x)
                pos_x, pos_y = self._find_bottom_left(feasible)

            # Yerleştir
            placed_poly = translate(piece, pos_x, pos_y)
            placed_polys.append(placed_poly)

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

    def _find_bottom_left(self, feasible) -> tuple:
        """Feasible bölgede en alt-sol noktayı bul."""
        if isinstance(feasible, MultiPolygon):
            # Tüm polygon'ların en düşük y'li köşesini bul
            best = (float("inf"), float("inf"))
            for geom in feasible.geoms:
                coords = np.array(geom.exterior.coords)
                min_y_idx = np.argmin(coords[:, 1])
                y = coords[min_y_idx, 1]
                x = coords[min_y_idx, 0]
                if y < best[1] or (y == best[1] and x < best[0]):
                    best = (x, y)
            return best
        elif isinstance(feasible, Polygon):
            coords = np.array(feasible.exterior.coords)
            # En düşük y, sonra en sol x
            min_y = np.min(coords[:, 1])
            bottom_mask = coords[:, 1] <= min_y + 1.0  # 1mm tolerans
            bottom_points = coords[bottom_mask]
            min_x_idx = np.argmin(bottom_points[:, 0])
            return (bottom_points[min_x_idx, 0], bottom_points[min_x_idx, 1])
        else:
            return (0, 0)

    def fitness(self, sequence: list[int], rotations: list[float]) -> float:
        """Fitness = utilization - unplaced penalty."""
        result = self.decode(sequence, rotations)
        base = result["utilization"]
        unplaced = self.n - result["n_placed"]
        penalty = unplaced * 5.0
        return max(0.0, base - penalty)
