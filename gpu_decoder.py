"""GPU-accelerated Skyline-BLF decoder v2.0 — TRUE batch parallelism.

Pop=2000, iter=50000 hedefi icin optimize edildi.
Tum bireyler AYNI ANDA islenir — batch tensor ops, minimal Python loops.

A100 hedefi: 100M fitness evaluation birkaç saatte.
CPU fallback: device='cpu' ile de calisir (yavas ama calisan).

Architecture:
  Outer loop: N pieces (sequential — inherent dependency)
  Inner: B individuals processed simultaneously via batched tensor ops.
  For B=2000, N=13: only 13 piece-steps, each fully batched on GPU.
"""
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def _scanline_rasterize(coords: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fast scanline rasterization — matches CPU decoder v3."""
    mask = np.zeros((h, w), dtype=np.uint8)
    n = len(coords) - 1
    y0 = coords[:-1, 1]
    y1 = coords[1:, 1]
    x0 = coords[:-1, 0]
    x1 = coords[1:, 0]

    for row in range(h):
        y = row + 0.5
        cross = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        if not np.any(cross):
            continue
        dy = y1[cross] - y0[cross]
        t = (y - y0[cross]) / dy
        x_int = x0[cross] + t * (x1[cross] - x0[cross])
        x_int.sort()
        for i in range(0, len(x_int) - 1, 2):
            xs = max(0, int(x_int[i]))
            xe = min(w, int(np.ceil(x_int[i + 1])))
            if xs < xe:
                mask[row, xs:xe] = 1
    return mask


class GPUDecoder:
    """Batch Skyline-BLF decoder — TRUE GPU parallelism for B=2000+.

    Key insight: all B individuals share piece bitmaps. At each piece step,
    batch all B skyline scans, collision checks, and placements into
    GPU tensor operations.
    """

    def __init__(self, pieces: list[dict], bin_width: float = 1500.0,
                 resolution: float = 3.0, device: str = 'cuda'):
        self.original_pieces = pieces
        self.bin_width = bin_width
        self.res = resolution
        self.n = len(pieces)
        self.bin_w = int(bin_width / resolution)

        if device == 'cuda' and not torch.cuda.is_available():
            print("[GPUDecoder] CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)

        self.piece_areas = torch.tensor(
            [p["area"] for p in pieces], dtype=torch.float32, device=self.device)

        self.max_h = int(sum(
            max(p["width"], p["height"]) for p in self.original_pieces
        ) * 1.5 / self.res)

        self._bitmap_cache = {}
        self._precompute_bitmaps()

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def _rasterize_np(self, poly: Polygon, rotation: float) -> np.ndarray:
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
        gc = coords.copy()
        gc[:, 0] = (coords[:, 0] - b[0]) / res
        gc[:, 1] = (coords[:, 1] - b[1]) / res
        return _scanline_rasterize(gc, w, h)

    def _precompute_bitmaps(self):
        for i, p in enumerate(self.original_pieces):
            self._bitmap_cache[i] = {}
            for angle in [0, 90, 180, 270]:
                bmp = self._rasterize_np(p["polygon"], float(angle))
                self._bitmap_cache[i][angle] = torch.from_numpy(bmp).to(
                    dtype=torch.bool, device=self.device)

    def _get_bitmap(self, piece_idx: int, rotation: float) -> torch.Tensor:
        akey = int(round(rotation % 360, 0)) % 360
        cache = self._bitmap_cache[piece_idx]
        if akey in cache:
            return cache[akey]
        bmp = self._rasterize_np(
            self.original_pieces[piece_idx]["polygon"], float(akey))
        t = torch.from_numpy(bmp).to(dtype=torch.bool, device=self.device)
        cache[akey] = t
        return t

    # ------------------------------------------------------------------
    # CORE: batch_fitness — B=2000 individuals, TRUE parallel
    # ------------------------------------------------------------------

    @torch.no_grad()
    def batch_fitness(self, sequences: list[list[int]],
                      rotations: list[list[float]]) -> list[float]:
        """Evaluate B solutions in TRUE parallel.

        Memory: B * max_h * bin_w bits (bool). For B=2000, max_h=500, bin_w=500:
        ~500MB — fits comfortably on A100 (80GB) or even 8GB consumer GPU.
        """
        B = len(sequences)
        if B == 0:
            return []

        max_h = self.max_h
        bin_w = self.bin_w
        dev = self.device
        n_pieces = self.n

        # Batch state
        canvases = torch.zeros((B, max_h, bin_w), dtype=torch.bool, device=dev)
        skylines = torch.zeros((B, bin_w), dtype=torch.int32, device=dev)
        placed_mask = torch.zeros((B, n_pieces), dtype=torch.bool, device=dev)

        # Sequences/rotations as tensors
        seq_t = torch.tensor(sequences, dtype=torch.long, device=dev)     # [B, N]
        rot_t = torch.tensor(rotations, dtype=torch.float32, device=dev)  # [B, N]

        for step in range(n_pieces):
            # piece_ids[b] = which piece individual b places at this step
            piece_ids = seq_t[:, step]  # [B]

            # Rotation for each individual's piece at this step
            piece_rots = torch.gather(
                rot_t, 1, piece_ids.unsqueeze(1)).squeeze(1)  # [B]
            angle_keys = (piece_rots.round().long() % 360)    # [B]

            # Group by (piece, angle) to share bitmaps
            combo_keys = piece_ids * 360 + angle_keys  # [B]
            unique_combos, inverse = torch.unique(combo_keys, return_inverse=True)

            for gi, combo in enumerate(unique_combos):
                combo_val = combo.item()
                pid = combo_val // 360
                akey = combo_val % 360

                # Members of this group
                member_idx = torch.where(inverse == gi)[0]
                M = len(member_idx)
                if M == 0:
                    continue

                bmp = self._get_bitmap(pid, float(akey))
                bh, bw_p = bmp.shape

                if bw_p > bin_w:
                    alt = (akey + 90) % 360
                    bmp = self._get_bitmap(pid, float(alt))
                    bh, bw_p = bmp.shape
                    if bw_p > bin_w:
                        continue

                x_range = bin_w - bw_p + 1
                if x_range <= 0:
                    continue

                # --- FULLY BATCHED skyline scan for M members ---
                m_sky = skylines[member_idx]  # [M, bin_w]

                # Sliding window max: base_y for all x positions
                # [M, x_range, bw_p] -> max -> [M, x_range]
                sky_win = m_sky.unfold(1, bw_p, 1)
                base_ys_all = sky_win.max(dim=2).values  # [M, x_range]

                tops_all = base_ys_all + bh  # [M, x_range]
                valid_all = tops_all < max_h  # [M, x_range]

                # For each member, pick the first valid x with lowest top
                # Set invalid to huge value
                INF = max_h * 2
                tops_masked = torch.where(valid_all, tops_all,
                                          torch.full_like(tops_all, INF))

                # Find the x with minimum top for each member (greedy first-fit)
                min_tops, best_x_candidates = tops_masked.min(dim=1)  # [M], [M]

                # Members with no valid position at all
                has_valid = (min_tops < max_h)

                # For members with valid positions, we need collision check.
                # BATCH collision: extract [M, bh, bw_p] regions at (best_x, base_y)
                # and check overlap with bmp.
                #
                # Strategy: try best_x first. If collision, try next candidates.
                # For speed, we use a multi-pass approach (max 4 passes).

                remaining = has_valid.clone()  # [M] — members still needing placement
                final_x = torch.full((M,), -1, dtype=torch.long, device=dev)
                final_y = torch.full((M,), -1, dtype=torch.long, device=dev)

                for pass_num in range(4):
                    if not remaining.any():
                        break

                    r_idx = torch.where(remaining)[0]  # active member indices within M

                    if pass_num == 0:
                        # First pass: try argmin(top) position
                        try_x = best_x_candidates[r_idx]  # x for each active member
                    else:
                        # Subsequent passes: mask out previous best x (vectorized)
                        prev_x = torch.where(
                            final_x[r_idx] >= 0,
                            final_x[r_idx],
                            best_x_candidates[r_idx])
                        tops_masked[r_idx, prev_x] = INF
                        new_min_tops, new_best_x = tops_masked[r_idx].min(dim=1)
                        # Filter out members with no more valid options
                        still_ok = new_min_tops < max_h
                        if not still_ok.any():
                            break
                        r_idx = r_idx[still_ok]
                        try_x = new_best_x[still_ok]

                    if len(r_idx) == 0:
                        break

                    try_y = base_ys_all[r_idx, try_x]  # [R]

                    # Batch collision check: for each active member, extract region
                    # and test overlap with bmp
                    # Use advanced indexing to extract [R, bh, bw_p] regions
                    R = len(r_idx)
                    batch_indices = member_idx[r_idx]  # indices into full B

                    # Build row/col indices for gathering
                    # canvas[batch_indices[r], try_y[r]:try_y[r]+bh, try_x[r]:try_x[r]+bw_p]
                    row_offsets = torch.arange(bh, device=dev).unsqueeze(0)  # [1, bh]
                    col_offsets = torch.arange(bw_p, device=dev).unsqueeze(0)  # [1, bw_p]

                    row_idx = try_y.unsqueeze(1) + row_offsets  # [R, bh]
                    col_idx = try_x.unsqueeze(1) + col_offsets  # [R, bw_p]

                    # Bounds check
                    row_ok = (row_idx[:, -1] < max_h)  # [R]
                    col_ok = (col_idx[:, -1] < bin_w)  # [R]
                    bounds_ok = row_ok & col_ok

                    if bounds_ok.any():
                        ok_idx = torch.where(bounds_ok)[0]

                        # Extract regions using batched indexing
                        bi = batch_indices[ok_idx]                # [K]
                        ry = row_idx[ok_idx]                      # [K, bh]
                        cx = col_idx[ok_idx]                      # [K, bw_p]

                        # Build [K, bh, bw_p] region by gathering
                        # For each k: region[k] = canvas[bi[k], ry[k,:], :][:, cx[k,:]]
                        # Efficient: use direct indexing with expand
                        K = len(ok_idx)
                        bi_exp = bi.view(K, 1, 1).expand(K, bh, bw_p)
                        ry_exp = ry.unsqueeze(2).expand(K, bh, bw_p)
                        cx_exp = cx.unsqueeze(1).expand(K, bh, bw_p)

                        regions = canvases[bi_exp, ry_exp, cx_exp]  # [K, bh, bw_p]

                        # Collision: overlap with bmp
                        bmp_exp = bmp.unsqueeze(0)  # [1, bh, bw_p]
                        collisions = (regions & bmp_exp).view(K, -1).any(dim=1)  # [K]

                        # Members with no collision → place them
                        no_coll = ~collisions
                        if no_coll.any():
                            place_in_ok = ok_idx[no_coll]
                            place_in_r = r_idx[place_in_ok]
                            final_x[place_in_r] = try_x[place_in_ok]
                            final_y[place_in_r] = try_y[place_in_ok]
                            remaining[place_in_r] = False

                    # Members that failed bounds check stay in remaining

                # Fallback: members still not placed — put at skyline top
                still_remaining = remaining & has_valid
                if still_remaining.any():
                    fb_idx = torch.where(still_remaining)[0]
                    fb_y = m_sky[fb_idx].max(dim=1).values  # [F]
                    fb_ok = (fb_y + bh) < max_h
                    if fb_ok.any():
                        fb_place = fb_idx[fb_ok]
                        final_x[fb_place] = 0
                        final_y[fb_place] = fb_y[fb_ok]
                        remaining[fb_place] = False

                # === Apply placements to canvas + skyline (BATCH) ===
                to_place = (final_x >= 0)
                if to_place.any():
                    p_idx = torch.where(to_place)[0]  # indices within M
                    p_bi = member_idx[p_idx]           # indices into B
                    p_x = final_x[p_idx]
                    p_y = final_y[p_idx]
                    P = len(p_idx)

                    # Batch stamp: build index arrays for all P placements at once
                    row_off = torch.arange(bh, device=dev)    # [bh]
                    col_off = torch.arange(bw_p, device=dev)  # [bw_p]

                    # [P, bh]
                    all_rows = p_y.unsqueeze(1) + row_off.unsqueeze(0)
                    # [P, bw_p]
                    all_cols = p_x.unsqueeze(1) + col_off.unsqueeze(0)

                    # Expand to [P, bh, bw_p] for 3D indexing
                    bi_3d = p_bi.view(P, 1, 1).expand(P, bh, bw_p)
                    ry_3d = all_rows.unsqueeze(2).expand(P, bh, bw_p)
                    cx_3d = all_cols.unsqueeze(1).expand(P, bh, bw_p)

                    # OR-stamp bitmap onto all P canvases at once
                    canvases[bi_3d, ry_3d, cx_3d] |= bmp.unsqueeze(0)

                    # Batch skyline update: piece_top = p_y + bh
                    # For each placement, skyline[bi, x:x+bw_p] = max(sky, top)
                    sky_bi = p_bi.unsqueeze(1).expand(P, bw_p)  # [P, bw_p]
                    piece_tops = (p_y + bh).unsqueeze(1).expand(P, bw_p)  # [P, bw_p]
                    # Scatter max: for each (bi, col), set skyline to max
                    # Use scatter_reduce with 'amax' (PyTorch 2.0+)
                    flat_bi = sky_bi.reshape(-1)       # [P * bw_p]
                    flat_col = all_cols.reshape(-1)    # [P * bw_p]
                    flat_top = piece_tops.reshape(-1)  # [P * bw_p]

                    # Convert to linear index for scatter
                    linear_idx = flat_bi * bin_w + flat_col  # [P * bw_p]
                    flat_sky = skylines.view(-1)  # [B * bin_w]
                    flat_sky.scatter_reduce_(
                        0, linear_idx.long(), flat_top,
                        reduce='amax', include_self=True)

                    placed_mask[p_bi, pid] = True

        # === Compute fitness (fully vectorized) ===
        used_h = skylines.max(dim=1).values.float()         # [B]
        used_length = used_h * self.res                      # [B]
        placed_areas = placed_mask.float() @ self.piece_areas  # [B]
        bin_area = torch.clamp(self.bin_width * used_length, min=1.0)
        utilization = placed_areas / bin_area * 100.0
        n_placed = placed_mask.sum(dim=1).float()
        penalty = (self.n - n_placed) * 5.0
        fitness = torch.clamp(utilization - penalty, min=0.0)

        return fitness.tolist()

    # ------------------------------------------------------------------
    # Single-solution API (compatible with CPU decoder)
    # ------------------------------------------------------------------

    def fitness(self, sequence: list[int], rotations: list[float]) -> float:
        return self.batch_fitness([sequence], [rotations])[0]

    def decode(self, sequence: list[int], rotations: list[float]) -> dict:
        """Full decode with placement geometry — for export."""
        max_h = self.max_h
        bin_w = self.bin_w

        canvas = torch.zeros((max_h, bin_w), dtype=torch.bool, device=self.device)
        skyline = torch.zeros(bin_w, dtype=torch.int32, device=self.device)
        placed = []

        for idx in sequence:
            rot = rotations[idx]
            bmp = self._get_bitmap(idx, rot)
            bh, bw_p = bmp.shape

            if bw_p > bin_w:
                rot = (rot + 90) % 360
                bmp = self._get_bitmap(idx, rot)
                bh, bw_p = bmp.shape
                if bw_p > bin_w:
                    continue

            x_range = bin_w - bw_p + 1
            if x_range <= 0:
                continue

            sky_win = skyline.unfold(0, bw_p, 1)
            base_ys = sky_win.max(dim=1).values
            tops = base_ys + bh
            valid_mask = tops < max_h
            valid_xs = torch.where(valid_mask)[0]

            if len(valid_xs) == 0:
                best_y = int(skyline.max().item())
                best_x = 0
                if best_y + bh >= max_h:
                    continue
            else:
                valid_tops = tops[valid_xs]
                sort_order = torch.argsort(valid_tops)
                sorted_xs = valid_xs[sort_order]

                best_x = -1
                best_y = -1
                for ci in range(min(len(sorted_xs), 500)):
                    x = sorted_xs[ci].item()
                    by = int(base_ys[x].item())
                    region = canvas[by:by + bh, x:x + bw_p]
                    if region.shape[0] != bh or region.shape[1] != bw_p:
                        continue
                    if not torch.any(region & bmp).item():
                        best_x = x
                        best_y = by
                        break

                if best_x < 0:
                    best_y = int(skyline.max().item())
                    best_x = 0
                    if best_y + bh >= max_h:
                        continue

            canvas[best_y:best_y + bh, best_x:best_x + bw_p] |= bmp
            piece_top = best_y + bh
            skyline[best_x:best_x + bw_p] = torch.clamp(
                skyline[best_x:best_x + bw_p], min=piece_top)

            placed.append({
                "piece_id": idx,
                "x": best_x * self.res,
                "y": best_y * self.res,
                "rotation": rot,
                "polygon": self._get_placed_polygon(
                    idx, rot, best_x * self.res, best_y * self.res),
            })

        if not placed:
            return {"placements": [], "used_length": 0, "utilization": 0.0,
                    "total_piece_area": 0, "bin_area": 0, "n_placed": 0}

        used_length = float(skyline.max().item()) * self.res
        total_piece_area = sum(
            self.original_pieces[p["piece_id"]]["area"] for p in placed)
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
