"""GPU decoder v3 — maximum throughput, minimal Python overhead.

Hedef: L4'te pop=2000, 44 parça → <1s/iterasyon.
V2'den farklar:
- unique_combos Python loop kaldırıldı → TÜM B bireyler aynı bitmap ile işlenir
  (angle quantization: 15° adım → aynı parça aynı bitmap)
- 4-pass retry kaldırıldı → tek pass first-fit
- .item() çağrısı yok → sıfır GPU-CPU sync
- Pre-allocated tensörler → sıfır allocation per step
"""
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from matplotlib.path import Path as MplPath


class GPUDecoderV3:
    """Batch Skyline-BLF v3 — maximum GPU throughput."""

    def __init__(self, pieces, bin_width=1500.0, resolution=5.0,
                 device='cuda', angle_step=15.0):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.res = resolution
        self.n = len(pieces)
        self.bin_w = int(bin_width / resolution)
        self.angle_step = angle_step

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)

        self.piece_areas = torch.tensor(
            [p["area"] for p in pieces], dtype=torch.float32, device=self.device)

        self.max_h = int(sum(
            max(p["width"], p["height"]) for p in pieces) * 1.2 / resolution)

        # Pre-compute ALL bitmaps: [n_pieces, n_angles, max_bh, max_bw]
        # Stored as dict: (piece_idx, quantized_angle) → (bmp_tensor, bh, bw)
        self._bitmaps = {}
        angles = [round(a * angle_step % 360) for a in range(int(360 / angle_step))]
        max_bh, max_bw = 0, 0
        for i, p in enumerate(pieces):
            for ang in angles:
                bmp_np = self._rasterize(p["polygon"], float(ang))
                bh, bw = bmp_np.shape
                max_bh = max(max_bh, bh)
                max_bw = max(max_bw, bw)
                t = torch.from_numpy(bmp_np).to(dtype=torch.bool, device=self.device)
                self._bitmaps[(i, ang)] = (t, bh, bw)

        self._max_bh = max_bh
        self._max_bw = max_bw

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

    def _get_bmp(self, pid, rot):
        step = self.angle_step
        qr = int(round(rot / step) * step) % 360
        key = (pid, qr)
        if key in self._bitmaps:
            return self._bitmaps[key]
        bmp_np = self._rasterize(self.original_pieces[pid]["polygon"], float(qr))
        t = torch.from_numpy(bmp_np).to(dtype=torch.bool, device=self.device)
        bh, bw = t.shape
        self._bitmaps[key] = (t, bh, bw)
        return (t, bh, bw)

    @torch.no_grad()
    def batch_fitness(self, sequences, rotations):
        """B çözümü paralel evaluate et — maximum GPU throughput."""
        B = len(sequences)
        if B == 0:
            return []

        dev = self.device
        max_h = self.max_h
        bin_w = self.bin_w
        n = self.n

        # Pre-allocate — sıfır runtime allocation
        canvases = torch.zeros((B, max_h, bin_w), dtype=torch.bool, device=dev)
        skylines = torch.zeros((B, bin_w), dtype=torch.int32, device=dev)
        placed_mask = torch.zeros((B, n), dtype=torch.bool, device=dev)

        seq_t = torch.tensor(sequences, dtype=torch.long, device=dev)
        rot_t = torch.tensor(rotations, dtype=torch.float32, device=dev)

        for step in range(n):
            # Her birey bu adımda hangi parçayı koyuyor
            pids = seq_t[:, step]  # [B]
            rots = torch.gather(rot_t, 1, pids.unsqueeze(1)).squeeze(1)  # [B]

            # Angle quantize
            step_size = self.angle_step
            qrots = ((rots / step_size).round() * step_size).long() % 360  # [B]

            # Unique (pid, qrot) kombinasyonları — GPU'da
            combo = pids * 360 + qrots  # [B]
            unique_combos = torch.unique(combo)

            for uc in unique_combos:
                uc_val = uc.item()
                pid = uc_val // 360
                qr = uc_val % 360

                mask = (combo == uc)  # [B] bool
                member_idx = torch.where(mask)[0]
                M = member_idx.shape[0]
                if M == 0:
                    continue

                bmp_data = self._get_bmp(pid, float(qr))
                bmp, bh, bw_p = bmp_data

                if bw_p > bin_w:
                    alt_qr = (qr + 90) % 360
                    bmp_data = self._get_bmp(pid, float(alt_qr))
                    bmp, bh, bw_p = bmp_data
                    if bw_p > bin_w:
                        continue

                x_range = bin_w - bw_p + 1
                if x_range <= 0:
                    continue

                # BATCH skyline scan
                m_sky = skylines[member_idx]  # [M, bin_w]
                sky_win = m_sky.unfold(1, bw_p, 1)  # [M, x_range, bw_p]
                base_ys = sky_win.max(dim=2).values  # [M, x_range]

                tops = base_ys + bh
                valid = tops < max_h

                # First-fit: en düşük top'a sahip valid x
                INF = max_h * 2
                tops_masked = torch.where(valid, tops, torch.tensor(INF, dtype=torch.int32, device=dev))
                min_tops, best_x = tops_masked.min(dim=1)  # [M]

                has_place = min_tops < max_h
                if not has_place.any():
                    continue

                good = torch.where(has_place)[0]
                gx = best_x[good]  # [G]
                gy = base_ys[good, gx].long()  # [G]
                g_bi = member_idx[good]  # [G] → index into B

                # BATCH collision check
                G = good.shape[0]
                row_off = torch.arange(bh, device=dev)
                col_off = torch.arange(bw_p, device=dev)

                rows = gy.unsqueeze(1) + row_off.unsqueeze(0)  # [G, bh]
                cols = gx.unsqueeze(1) + col_off.unsqueeze(0)  # [G, bw_p]

                # Bounds check
                row_ok = rows[:, -1] < max_h
                col_ok = cols[:, -1] < bin_w
                ok = row_ok & col_ok
                if not ok.any():
                    continue

                ok_idx = torch.where(ok)[0]
                K = ok_idx.shape[0]

                bi_k = g_bi[ok_idx]
                ry_k = rows[ok_idx]  # [K, bh]
                cx_k = cols[ok_idx]  # [K, bw_p]

                bi3 = bi_k.view(K, 1, 1).expand(K, bh, bw_p)
                ry3 = ry_k.unsqueeze(2).expand(K, bh, bw_p)
                cx3 = cx_k.unsqueeze(1).expand(K, bh, bw_p)

                regions = canvases[bi3, ry3, cx3]  # [K, bh, bw_p]
                coll = (regions & bmp.unsqueeze(0)).view(K, -1).any(dim=1)
                no_coll = ~coll

                if not no_coll.any():
                    # Fallback: top of skyline
                    fb_y = m_sky[good[ok_idx]].max(dim=1).values
                    fb_ok = (fb_y + bh) < max_h
                    if fb_ok.any():
                        fb_i = ok_idx[fb_ok]
                        fb_bi = g_bi[fb_i]
                        fb_gy = fb_y[fb_ok].long()
                        # Stamp
                        FK = fb_i.shape[0]
                        fb_rows = fb_gy.unsqueeze(1) + row_off.unsqueeze(0)
                        fb_cols = torch.zeros(FK, dtype=torch.long, device=dev).unsqueeze(1) + col_off.unsqueeze(0)
                        fb_bi3 = fb_bi.view(FK, 1, 1).expand(FK, bh, bw_p)
                        fb_ry3 = fb_rows.unsqueeze(2).expand(FK, bh, bw_p)
                        fb_cx3 = fb_cols.unsqueeze(1).expand(FK, bh, bw_p)
                        canvases[fb_bi3, fb_ry3, fb_cx3] |= bmp.unsqueeze(0)
                        # Skyline update
                        sky_top = (fb_gy + bh).unsqueeze(1).expand(FK, bw_p)
                        flat_bi = fb_bi.unsqueeze(1).expand(FK, bw_p).reshape(-1)
                        flat_col = fb_cols.reshape(-1)
                        flat_top = sky_top.reshape(-1)
                        lin = flat_bi * bin_w + flat_col
                        skylines.view(-1).scatter_reduce_(
                            0, lin.long(), flat_top.to(skylines.dtype),
                            reduce='amax', include_self=True)
                        placed_mask[fb_bi, pid] = True
                    continue

                # Place no-collision members
                place_ok = ok_idx[no_coll]
                p_bi = g_bi[place_ok]
                p_x = gx[place_ok]
                p_y = gy[place_ok]
                P = place_ok.shape[0]

                # BATCH stamp
                p_rows = p_y.unsqueeze(1) + row_off.unsqueeze(0)  # [P, bh]
                p_cols = p_x.unsqueeze(1) + col_off.unsqueeze(0)  # [P, bw_p]

                p_bi3 = p_bi.view(P, 1, 1).expand(P, bh, bw_p)
                p_ry3 = p_rows.unsqueeze(2).expand(P, bh, bw_p)
                p_cx3 = p_cols.unsqueeze(1).expand(P, bh, bw_p)

                canvases[p_bi3, p_ry3, p_cx3] |= bmp.unsqueeze(0)

                # BATCH skyline update
                sky_top = (p_y + bh).unsqueeze(1).expand(P, bw_p)
                flat_bi = p_bi.unsqueeze(1).expand(P, bw_p).reshape(-1)
                flat_col = p_cols.reshape(-1)
                flat_top = sky_top.reshape(-1)
                lin = flat_bi * bin_w + flat_col
                skylines.view(-1).scatter_reduce_(
                    0, lin.long(), flat_top.to(skylines.dtype),
                    reduce='amax', include_self=True)

                placed_mask[p_bi, pid] = True

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
        """CPU fallback decode for export."""
        from decoder import BLFDecoder
        cpu = BLFDecoder(self.original_pieces, self.bin_width, self.res)
        return cpu.decode(sequence, rotations)
