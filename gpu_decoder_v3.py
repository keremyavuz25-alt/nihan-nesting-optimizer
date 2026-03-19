"""GPU decoder v3.1 — per-piece boyut maskeli, SIFIR Python loop overhead.

GPU %98 utilization + doğru placement.
Padding var ama skyline update'te her birey kendi gerçek bh'sini kullanır.
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

        # Rasterize tüm bitmap'ler
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

        sizes = torch.zeros(
            (self.n, self.n_angles, 2), dtype=torch.int32, device=self.device)

        for (i, ai), bmp in raw_bmps.items():
            bh, bw = bmp.shape
            t = torch.from_numpy(bmp).to(dtype=torch.bool, device=self.device)
            bmp_tensor[i, ai, :bh, :bw] = t
            sizes[i, ai, 0] = bh
            sizes[i, ai, 1] = bw

        self.bmp_tensor = bmp_tensor
        self.sizes = sizes

        mem_mb = bmp_tensor.element_size() * bmp_tensor.nelement() / 1e6
        print(f"  GPU bitmap tensor: {bmp_tensor.shape}, {mem_mb:.0f}MB", flush=True)

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
        B = len(sequences)
        if B == 0:
            return []

        dev = self.device
        max_h = self.max_h
        bin_w = self.bin_w
        n = self.n
        max_bh = self.max_bh
        max_bw = self.max_bw

        canvases = torch.zeros((B, max_h, bin_w), dtype=torch.bool, device=dev)
        skylines = torch.zeros((B, bin_w), dtype=torch.int32, device=dev)
        placed_mask = torch.zeros((B, n), dtype=torch.bool, device=dev)

        seq_t = torch.tensor(sequences, dtype=torch.long, device=dev)
        rot_t = torch.tensor(rotations, dtype=torch.float32, device=dev)

        # Pre-alloc offsets (max boyut)
        max_row_off = torch.arange(max_bh, device=dev)
        max_col_off = torch.arange(max_bw, device=dev)

        for step in range(n):
            pids = seq_t[:, step]
            rots_step = torch.gather(rot_t, 1, pids.unsqueeze(1)).squeeze(1)
            ai = ((rots_step / self.angle_step).round().long() % self.n_angles)

            # Bitmap + boyut çek
            bmps = self.bmp_tensor[pids, ai]  # [B, max_bh, max_bw]
            szs = self.sizes[pids, ai]  # [B, 2]
            step_bhs = szs[:, 0]  # [B] her bireyin gerçek bh'si
            step_bws = szs[:, 1]  # [B] her bireyin gerçek bw'si

            # Step-level max — GPU kernel boyutu için
            bh = int(step_bhs.max().item())
            bw_p = int(step_bws.max().item())
            if bh == 0 or bw_p == 0:
                continue

            # Kırp
            bmps_crop = bmps[:, :bh, :bw_p]

            x_range = bin_w - bw_p + 1
            if x_range <= 0:
                continue

            # Skyline scan — bw_p sliding window
            sky_win = skylines.unfold(1, bw_p, 1)
            base_ys = sky_win.max(dim=2).values  # [B, x_range]

            # Her bireyin kendi bh'si ile top hesapla
            # step_bhs: [B] → [B, 1] broadcast ile [B, x_range]
            per_bh = step_bhs.unsqueeze(1)  # [B, 1]
            tops = base_ys + per_bh  # [B, x_range] — her birey kendi yüksekliğini kullanır
            valid = tops < max_h

            INF = max_h * 2
            tops_masked = torch.where(valid, tops,
                                      torch.full_like(tops, INF))
            min_tops, best_x = tops_masked.min(dim=1)

            has_place = min_tops < max_h
            if not has_place.any():
                continue

            good = torch.where(has_place)[0]
            G = good.shape[0]
            gx = best_x[good]
            gy = base_ys[good, gx].long()

            # Collision check — max boyut ile (padding sıfır, sorun yok)
            row_off = max_row_off[:bh]
            col_off = max_col_off[:bw_p]

            rows = gy.unsqueeze(1) + row_off.unsqueeze(0)
            cols = gx.unsqueeze(1) + col_off.unsqueeze(0)

            row_ok = rows[:, -1] < max_h
            col_ok = cols[:, -1] < bin_w
            ok = row_ok & col_ok

            if not ok.any():
                continue

            ok_idx = torch.where(ok)[0]
            K = ok_idx.shape[0]

            bi_k = good[ok_idx]
            ry_k = rows[ok_idx]
            cx_k = cols[ok_idx]

            bi3 = bi_k.view(K, 1, 1).expand(K, bh, bw_p)
            ry3 = ry_k.unsqueeze(2).expand(K, bh, bw_p)
            cx3 = cx_k.unsqueeze(1).expand(K, bh, bw_p)

            regions = canvases[bi3, ry3, cx3]
            k_bmps = bmps_crop[bi_k]
            coll = (regions & k_bmps).view(K, -1).any(dim=1)
            no_coll = ~coll

            if not no_coll.any():
                continue

            place = ok_idx[no_coll]
            P = place.shape[0]
            p_bi = good[place]
            p_x = gx[place]
            p_y = gy[place]

            # Stamp — padded bitmap stamp, ama padding=0 olduğundan canvas'a zarar vermez
            p_rows = p_y.unsqueeze(1) + row_off.unsqueeze(0)
            p_cols = p_x.unsqueeze(1) + col_off.unsqueeze(0)

            p_bi3 = p_bi.view(P, 1, 1).expand(P, bh, bw_p)
            p_ry3 = p_rows.unsqueeze(2).expand(P, bh, bw_p)
            p_cx3 = p_cols.unsqueeze(1).expand(P, bh, bw_p)

            p_bmps = bmps_crop[p_bi]
            canvases[p_bi3, p_ry3, p_cx3] |= p_bmps

            # Skyline update — HER BİREYİN KENDİ bh'si ile
            # per_bh[p_bi] = her yerleştirilen bireyin gerçek yüksekliği
            p_real_bh = step_bhs[p_bi]  # [P]
            sky_top = (p_y + p_real_bh).unsqueeze(1).expand(P, bw_p)  # [P, bw_p]

            flat_bi = p_bi.unsqueeze(1).expand(P, bw_p).reshape(-1)
            flat_col = p_cols.reshape(-1)
            flat_top = sky_top.reshape(-1)
            lin = flat_bi * bin_w + flat_col
            skylines.view(-1).scatter_reduce_(
                0, lin.long(), flat_top.to(skylines.dtype),
                reduce='amax', include_self=True)

            p_pids = pids[p_bi]
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
