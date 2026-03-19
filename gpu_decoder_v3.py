"""GPU decoder v3 — padded bitmap tensor, SIFIR Python loop (piece step hariç).

Tüm bitmap'ler [n_pieces, n_angles, max_bh, max_bw] tensoründe.
unique_combos loop KALKTI. Tek batched indexing ile B bitmap çekilir.
GPU utilization hedefi: %60-80.
"""
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from matplotlib.path import Path as MplPath


class GPUDecoderV3:

    def __init__(self, pieces, bin_width=1500.0, resolution=5.0,
                 device='cuda', angle_step=15.0):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.res = resolution
        self.n = len(pieces)
        self.bin_w = int(bin_width / resolution)
        self.angle_step = angle_step
        self.n_angles = int(360 / angle_step)

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)

        self.piece_areas = torch.tensor(
            [p["area"] for p in pieces], dtype=torch.float32, device=self.device)

        self.max_h = int(sum(
            max(p["width"], p["height"]) for p in pieces) * 1.2 / resolution)

        # Tüm bitmap'leri rasterize et ve max boyut bul
        raw_bmps = {}
        max_bh, max_bw = 1, 1
        angles = list(range(self.n_angles))

        for i, p in enumerate(pieces):
            for ai in angles:
                ang = round(ai * angle_step) % 360
                bmp = self._rasterize(p["polygon"], float(ang))
                bh, bw = bmp.shape
                max_bh = max(max_bh, bh)
                max_bw = max(max_bw, bw)
                raw_bmps[(i, ai)] = bmp

        self.max_bh = max_bh
        self.max_bw = max_bw

        # Padded bitmap tensor: [n_pieces, n_angles, max_bh, max_bw]
        bmp_tensor = torch.zeros(
            (self.n, self.n_angles, max_bh, max_bw),
            dtype=torch.bool, device=self.device)

        # Gerçek boyutlar: [n_pieces, n_angles, 2] → (bh, bw)
        sizes = torch.zeros(
            (self.n, self.n_angles, 2), dtype=torch.int32, device=self.device)

        for (i, ai), bmp in raw_bmps.items():
            bh, bw = bmp.shape
            t = torch.from_numpy(bmp).to(dtype=torch.bool, device=self.device)
            bmp_tensor[i, ai, :bh, :bw] = t
            sizes[i, ai, 0] = bh
            sizes[i, ai, 1] = bw

        self.bmp_tensor = bmp_tensor  # [n, n_angles, max_bh, max_bw]
        self.sizes = sizes  # [n, n_angles, 2]

        print(f"  GPU bitmap tensor: {bmp_tensor.shape}, "
              f"{bmp_tensor.element_size() * bmp_tensor.nelement() / 1e6:.0f}MB", flush=True)

    def _rasterize(self, poly, rotation):
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
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        return path.contains_points(pts).reshape(h, w).astype(np.uint8)

    @torch.no_grad()
    def batch_fitness(self, sequences, rotations):
        """B çözümü paralel — SIFIR Python loop (piece step hariç)."""
        B = len(sequences)
        if B == 0:
            return []

        dev = self.device
        max_h = self.max_h
        bin_w = self.bin_w
        n = self.n
        max_bh = self.max_bh
        max_bw = self.max_bw
        angle_step = self.angle_step

        canvases = torch.zeros((B, max_h, bin_w), dtype=torch.bool, device=dev)
        skylines = torch.zeros((B, bin_w), dtype=torch.int32, device=dev)
        placed_mask = torch.zeros((B, n), dtype=torch.bool, device=dev)

        seq_t = torch.tensor(sequences, dtype=torch.long, device=dev)
        rot_t = torch.tensor(rotations, dtype=torch.float32, device=dev)

        # row_off ve col_off her step'te dinamik oluşturulacak

        for step in range(n):
            pids = seq_t[:, step]  # [B]
            rots = torch.gather(rot_t, 1, pids.unsqueeze(1)).squeeze(1)  # [B]

            # Angle quantize → angle index
            ai = ((rots / angle_step).round().long() % self.n_angles)  # [B]

            # TEK INDEXING — tüm B bireyler için bitmap ve boyut çek
            bmps = self.bmp_tensor[pids, ai]  # [B, max_bh, max_bw]
            szs = self.sizes[pids, ai]  # [B, 2] → (bh, bw)

            # Her birey farklı parça koyuyor → farklı boyutlar
            # GPU batch için bu step'teki MAX gerçek boyutu kullan
            # Padding sıfır → collision'da sorun yok, skyline'da max bh kadar artış
            step_bhs = szs[:, 0]  # [B]
            step_bws = szs[:, 1]  # [B]
            bh = int(step_bhs.max().item())  # bu step'teki en büyük yükseklik
            bw_p = int(step_bws.max().item())  # bu step'teki en büyük genişlik
            if bh == 0 or bw_p == 0:
                continue

            # Bitmap'leri gerçek boyuta kırp (pad'i kaldır)
            bmps = bmps[:, :bh, :bw_p]  # [B, bh, bw_p]

            x_range = bin_w - bw_p + 1
            if x_range <= 0:
                # Bazı parçalar sığmayabilir — bireysel kontrol gerekir
                # Basitleştirme: max bitmap ene sığmıyorsa skip
                continue

            # BATCH skyline scan — TÜM B bireyler, Python loop YOK
            sky_win = skylines.unfold(1, bw_p, 1)  # [B, x_range, bw_p]
            base_ys = sky_win.max(dim=2).values  # [B, x_range]

            tops = base_ys + bh
            valid = tops < max_h

            INF = max_h * 2
            tops_masked = torch.where(valid, tops,
                                      torch.tensor(INF, dtype=torch.int32, device=dev))
            min_tops, best_x = tops_masked.min(dim=1)  # [B]

            has_place = min_tops < max_h
            if not has_place.any():
                continue

            good = torch.where(has_place)[0]  # [G]
            G = good.shape[0]
            gx = best_x[good]
            gy = base_ys[good, gx].long()

            # BATCH collision check — tüm G bireyler tek seferde
            row_off = torch.arange(bh, device=dev)
            col_off = torch.arange(bw_p, device=dev)
            rows = gy.unsqueeze(1) + row_off.unsqueeze(0)  # [G, bh]
            cols = gx.unsqueeze(1) + col_off.unsqueeze(0)  # [G, bw_p]

            row_ok = rows[:, -1] < max_h
            col_ok = cols[:, -1] < bin_w
            ok = row_ok & col_ok

            if not ok.any():
                continue

            ok_idx = torch.where(ok)[0]
            K = ok_idx.shape[0]

            bi_k = good[ok_idx]  # [K] → index into B
            ry_k = rows[ok_idx]  # [K, max_bh]
            cx_k = cols[ok_idx]  # [K, max_bw]

            bi3 = bi_k.view(K, 1, 1).expand(K, bh, bw_p)
            ry3 = ry_k.unsqueeze(2).expand(K, bh, bw_p)
            cx3 = cx_k.unsqueeze(1).expand(K, bh, bw_p)

            regions = canvases[bi3, ry3, cx3]  # [K, max_bh, max_bw]

            # Her bireyin kendi bitmap'i ile collision check
            k_bmps = bmps[bi_k]  # [K, max_bh, max_bw]
            coll = (regions & k_bmps).view(K, -1).any(dim=1)
            no_coll = ~coll

            if not no_coll.any():
                continue

            # BATCH stamp — collision-free olanları yerleştir
            place = ok_idx[no_coll]
            P = place.shape[0]
            p_bi = good[place]
            p_x = gx[place]
            p_y = gy[place]

            p_rows = p_y.unsqueeze(1) + torch.arange(bh, device=dev).unsqueeze(0)
            p_cols = p_x.unsqueeze(1) + torch.arange(bw_p, device=dev).unsqueeze(0)

            p_bi3 = p_bi.view(P, 1, 1).expand(P, bh, bw_p)
            p_ry3 = p_rows.unsqueeze(2).expand(P, bh, bw_p)
            p_cx3 = p_cols.unsqueeze(1).expand(P, bh, bw_p)

            # Her bireyin kendi bitmap'i ile stamp
            p_bmps = bmps[p_bi]  # [P, max_bh, max_bw]
            canvases[p_bi3, p_ry3, p_cx3] |= p_bmps

            # BATCH skyline update
            sky_top = (p_y + bh).unsqueeze(1).expand(P, bw_p)
            flat_bi = p_bi.unsqueeze(1).expand(P, bw_p).reshape(-1)
            flat_col = p_cols.reshape(-1)
            flat_top = sky_top.reshape(-1)
            lin = flat_bi * bin_w + flat_col
            skylines.view(-1).scatter_reduce_(
                0, lin.long(), flat_top.to(skylines.dtype),
                reduce='amax', include_self=True)

            # Placed mask — step'teki parça id'leri
            p_pids = pids[p_bi]  # [P]
            placed_mask[p_bi, p_pids] = True

        # Fitness
        used_h = skylines.max(dim=1).values.float()
        used_length = used_h * self.res
        placed_areas = placed_mask.float() @ self.piece_areas
        bin_area = torch.clamp(self.bin_width * used_length, min=1.0)
        utilization = placed_areas / bin_area * 100.0
        n_placed = placed_mask.sum(dim=1).float()
        penalty = (self.n - n_placed) * 5.0
        fitness = torch.clamp(utilization - penalty, min=0.0)

        return fitness.tolist()

    def fitness(self, sequence, rotations):
        return self.batch_fitness([sequence], [rotations])[0]

    def decode(self, sequence, rotations):
        from decoder import BLFDecoder
        cpu = BLFDecoder(self.original_pieces, self.bin_width, self.res)
        return cpu.decode(sequence, rotations)
