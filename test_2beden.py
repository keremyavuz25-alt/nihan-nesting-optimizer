"""2 beden testi — 26 parçayı minimum kumaşa sıkıştır."""
from dxf_parser import load_dxf
from decoder import BLFDecoder
from algorithms import genetic_algorithm, ga_sa_hybrid, sparrow_search
import numpy as np
import time

# 1 beden yükle
pieces_1 = load_dxf('test.dxf')
print(f'Tek beden: {len(pieces_1)} parça, alan: {sum(p["area"] for p in pieces_1)/100:.0f} cm²')

# 2 beden — parçaları kopyala
pieces = []
for beden in range(2):
    for p in pieces_1:
        pieces.append({
            "id": len(pieces),
            "polygon": p["polygon"],
            "area": p["area"],
            "width": p["width"],
            "height": p["height"],
            "centroid": p["centroid"],
            "vertices": p["vertices"].copy(),
        })

n = len(pieces)
total_area = sum(p["area"] for p in pieces)
min_length = total_area / 1500  # teorik min (mm)
print(f'2 beden: {n} parça, alan: {total_area/100:.0f} cm²')
print(f'Teorik min uzunluk: {min_length:.0f}mm (%100 verimlilik)')
print()

# Decoder
decoder = BLFDecoder(pieces, bin_width=1500, resolution=5.0)

# Rastgele baseline
np.random.seed(42)
seq = np.random.permutation(n).tolist()
rots = [0.0] * n
baseline = decoder.decode(seq, rots)
print(f'Rastgele yerleştirme: {baseline["utilization"]:.1f}%, uzunluk: {baseline["used_length"]:.0f}mm')
print()

# 3 algoritma dene
for name, alg_fn in [
    ("GA", lambda: genetic_algorithm(decoder.fitness, n, pop_size=50, max_iter=300, verbose=False)),
    ("GA+SA", lambda: ga_sa_hybrid(decoder.fitness, n, pop_size=50, max_iter=300, verbose=False)),
    ("SSA", lambda: sparrow_search(decoder.fitness, n, pop_size=50, max_iter=300, verbose=False)),
]:
    np.random.seed(42)
    t0 = time.time()
    result = alg_fn()
    elapsed = time.time() - t0

    # Decode ile detay al
    best_seq, best_rots = result["best_solution"]
    layout = decoder.decode(best_seq, best_rots)

    print(f'{name}: {result["best_fitness"]:.1f}%, uzunluk: {layout["used_length"]:.0f}mm, '
          f'yerleşen: {layout["n_placed"]}/{n}, süre: {elapsed:.0f}s')

print(f'\n1 beden max: %54 (referans)')
print(f'2 beden hedef: >%54 (boşluklar dolacak)')
