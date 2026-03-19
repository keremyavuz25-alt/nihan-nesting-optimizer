"""Multi-beden test — aynı kalıbı N kere tekrarlayıp verimlilik ölçümü."""
from dxf_parser import load_dxf
from decoder import BLFDecoder
from algorithms import genetic_algorithm
import numpy as np
import time

pieces_1 = load_dxf('test.dxf')
print(f'Tek beden: {len(pieces_1)} parça')

for N_BEDEN in [1, 2, 4, 6, 8, 10]:
    # Parçaları N kere kopyala
    pieces = []
    for beden in range(N_BEDEN):
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
    decoder = BLFDecoder(pieces, bin_width=1500, resolution=5.0)

    # Hızlı GA — 100 iter, pop=30
    np.random.seed(42)
    t0 = time.time()
    result = genetic_algorithm(decoder.fitness, n, pop_size=30, max_iter=100, verbose=False)
    elapsed = time.time() - t0

    print(f'{N_BEDEN} beden ({n:>3d} parça): {result["best_fitness"]:.1f}%  ({elapsed:.1f}s)')
