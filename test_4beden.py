"""4 beden testi — tüm DXF'ler, GPU batch destekli."""
import glob, os, sys
import numpy as np
import time

# Unbuffered output — Colab'da anında görsün
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from dxf_parser import load_dxf
from decoder import BLFDecoder
from algorithms import ga_sa_hybrid

# GPU decoder varsa kullan
try:
    import torch
    from gpu_decoder import GPUDecoder
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU yok — CPU mode")
except ImportError:
    HAS_GPU = False
    print("torch yüklü değil — CPU mode")

BIN_WIDTH = 1500
N_BEDEN = 4
POP = 50
ITER = 200

dxf_dir = sys.argv[1] if len(sys.argv) > 1 else "dxf_samples"
dxf_files = sorted(glob.glob(os.path.join(dxf_dir, "*.dxf")))
if not dxf_files:
    dxf_files = ["test.dxf"]

print(f"\n{'='*70}")
print(f"4 BEDEN NESTING — {'GPU' if HAS_GPU else 'CPU'}")
print(f"{'='*70}")
print(f"Kumaş eni: {BIN_WIDTH}mm, Beden: {N_BEDEN}")
print(f"GA+SA: pop={POP}, iter={ITER}")
print()

results = []

for dxf_path in dxf_files:
    name = os.path.basename(dxf_path)

    pieces_1 = load_dxf(dxf_path, material="KUMAS")
    if not pieces_1:
        print(f"--- {name}: KUMAS parça yok, atlıyorum ---\n")
        continue

    n_1 = len(pieces_1)
    area_1 = sum(p["area"] for p in pieces_1)

    # 4 beden kopyala
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
    total_area = sum(p["area"] for p in pieces)
    min_length = total_area / BIN_WIDTH

    print(f"--- {name} ---")
    print(f"  1 beden: {n_1} KUMAS parça, {area_1/100:.0f} cm²")
    print(f"  4 beden: {n} parça, {total_area/100:.0f} cm²")
    print(f"  Teorik min: {min_length:.0f}mm ({min_length/1000:.2f}m)")

    # Decoder seç
    batch_fn = None
    if HAS_GPU:
        decoder = GPUDecoder(pieces, bin_width=BIN_WIDTH, resolution=5.0)
        batch_fn = decoder.batch_fitness
    else:
        decoder = BLFDecoder(pieces, bin_width=BIN_WIDTH, resolution=5.0)

    # Rastgele baseline
    np.random.seed(42)
    seq = np.random.permutation(n).tolist()
    rots = [0.0] * n
    baseline = decoder.decode(seq, rots)
    print(f"  Rastgele: {baseline['utilization']:.1f}%, {baseline['used_length']:.0f}mm", flush=True)

    # GA+SA with progress callback
    from progress import write_progress
    _t0_prog = time.time()
    _name_prog = name

    def _on_progress(it, best_fit):
        write_progress(_name_prog, it, ITER, best_fit, time.time() - _t0_prog)

    np.random.seed(42)
    t0 = time.time()
    result = ga_sa_hybrid(
        decoder.fitness, n, pop_size=POP, max_iter=ITER, verbose=True,
        batch_fitness_fn=batch_fn,
        progress_file=name,
    )
    elapsed = time.time() - t0

    best_seq, best_rots = result["best_solution"]
    layout = decoder.decode(best_seq, best_rots)
    fire_m = (layout['used_length'] - min_length) / 1000

    print(f"  GA+SA:   {result['best_fitness']:.1f}%, {layout['used_length']:.0f}mm ({layout['used_length']/1000:.2f}m), "
          f"yerleşen: {layout['n_placed']}/{n}, süre: {elapsed:.0f}s")
    print(f"  Fire:    {fire_m:.2f}m ({(1 - result['best_fitness']/100)*100:.1f}%)")
    print()

    results.append({
        "model": name,
        "parcalar_1beden": n_1,
        "parcalar_4beden": n,
        "teorik_min_mm": min_length,
        "rastgele_pct": baseline["utilization"],
        "gasa_pct": result["best_fitness"],
        "gasa_uzunluk_mm": layout["used_length"],
        "fire_m": fire_m,
        "sure_s": elapsed,
    })

# Özet tablo
print(f"\n{'='*80}")
print(f"{'MODEL':<25} {'PARÇA':>6} {'TEORİK':>8} {'RAST.':>6} {'GA+SA':>6} {'UZUNLUK':>9} {'FİRE':>7} {'SÜRE':>6}")
print(f"{'='*80}")
for r in results:
    print(f"{r['model']:<25} {r['parcalar_4beden']:>6} {r['teorik_min_mm']:>7.0f}mm {r['rastgele_pct']:>5.1f}% {r['gasa_pct']:>5.1f}% {r['gasa_uzunluk_mm']:>8.0f}mm {r['fire_m']:>6.2f}m {r['sure_s']:>5.0f}s")
print(f"{'='*80}")

if results:
    avg_util = np.mean([r["gasa_pct"] for r in results])
    avg_fire = np.mean([r["fire_m"] for r in results])
    print(f"\nOrtalama verimlilik: {avg_util:.1f}%")
    print(f"Ortalama fire: {avg_fire:.2f}m/pastal")
