"""Decoder hız testi — CPU vs GPU, single vs batch."""
from dxf_parser import load_dxf
from decoder import BLFDecoder
import time, numpy as np

try:
    from gpu_decoder import GPUDecoder
    HAS_GPU_DECODER = True
except ImportError:
    HAS_GPU_DECODER = False
    print("[UYARI] torch yüklü değil — GPU testleri atlanacak.\n"
          "        pip install torch ile yükleyip tekrar çalıştırın.\n")

pieces = load_dxf('test.dxf')
n = len(pieces)
print(f'{n} pieces loaded')

# ============================================================
# 1. CPU DECODER HIZ TESTİ
# ============================================================
print("\n" + "="*60)
print("1. CPU DECODER (BLFDecoder v3)")
print("="*60)

print("\nCache oluşturuluyor (res=3mm, 4 cardinal açı)...")
t0 = time.time()
cpu_dec = BLFDecoder(pieces, bin_width=1500, resolution=3.0)
cpu_init = time.time() - t0
print(f"Cache süresi: {cpu_init:.2f}s")

# Cardinal açı (cache hit)
np.random.seed(42)
seq = np.random.permutation(n).tolist()
rots = [0.0] * n
t0 = time.time()
result = cpu_dec.decode(seq, rots)
t1 = time.time()
print(f'\nCardinal (cache hit): {t1-t0:.4f}s, util={result["utilization"]:.2f}%, placed={result["n_placed"]}/{n}')

# Continuous açı (cache miss)
rots_cont = np.random.uniform(0, 360, size=n).tolist()
t0 = time.time()
result2 = cpu_dec.decode(seq, rots_cont)
t1 = time.time()
print(f'Continuous (cache miss): {t1-t0:.4f}s, util={result2["utilization"]:.2f}%, placed={result2["n_placed"]}/{n}')

# 50 decode benchmark
print(f'\n50 decode benchmark (continuous rotation)...')
np.random.seed(0)
t0 = time.time()
for i in range(50):
    s = np.random.permutation(n).tolist()
    r = np.random.uniform(0, 360, size=n).tolist()
    cpu_dec.decode(s, r)
cpu_50_time = time.time() - t0
cpu_avg = cpu_50_time / 50
print(f'50 decode: {cpu_50_time:.2f}s, avg={cpu_avg:.4f}s/decode')

# Fitness testi
f1 = cpu_dec.fitness(seq, rots)
print(f'\nFitness: {f1:.2f}%')


if not HAS_GPU_DECODER:
    # ============================================================
    # GPU DECODER YOK — Sadece CPU tahminleri
    # ============================================================
    print("\n" + "="*60)
    print("GPU DECODER TESTLERİ ATLANDI (torch yüklü değil)")
    print("="*60)

    def fmt_time(secs):
        if secs < 60:
            return f"{secs:.1f}s"
        elif secs < 3600:
            return f"{secs/60:.1f} dk"
        elif secs < 86400:
            return f"{secs/3600:.1f} saat"
        else:
            return f"{secs/86400:.1f} gün"

    total_evals = 2000 * 50000  # 100M
    cpu_est = cpu_avg * total_evals

    print(f"""
CPU-only tahminler (pop=2000, iter=50000, 100M eval):
  CPU BLF v3 sıralı: {cpu_avg*1000:.2f} ms/eval -> {fmt_time(cpu_est)}

  GPU testleri için: pip install torch
  Vertex AI A100'de test_speed.py tekrar çalıştırın.
""")

else:
    # ============================================================
    # 2. GPU DECODER — TEK FITNESS
    # ============================================================
    print("\n" + "="*60)
    print("2. GPU DECODER — TEK FITNESS (device=cpu fallback)")
    print("="*60)

    print("\nGPU decoder oluşturuluyor...")
    t0 = time.time()
    gpu_dec = GPUDecoder(pieces, bin_width=1500, resolution=3.0, device='cpu')
    gpu_init = time.time() - t0
    print(f"Init süresi: {gpu_init:.2f}s")
    print(f"Device: {gpu_dec.device}")

    # Tek fitness — doğrulama
    np.random.seed(42)
    seq = np.random.permutation(n).tolist()
    rots = [0.0] * n
    t0 = time.time()
    gpu_f = gpu_dec.fitness(seq, rots)
    gpu_single = time.time() - t0
    print(f'\nTek fitness: {gpu_single:.4f}s, fitness={gpu_f:.2f}%')

    # 50 sıralı tek fitness
    print(f'\n50 sıralı tek fitness...')
    np.random.seed(0)
    seqs_50 = [np.random.permutation(n).tolist() for _ in range(50)]
    rots_50 = [np.random.uniform(0, 360, size=n).tolist() for _ in range(50)]

    t0 = time.time()
    for s, r in zip(seqs_50, rots_50):
        gpu_dec.fitness(s, r)
    gpu_50_serial = time.time() - t0
    gpu_serial_avg = gpu_50_serial / 50
    print(f'50 sıralı: {gpu_50_serial:.2f}s, avg={gpu_serial_avg:.4f}s/decode')


    # ============================================================
    # 3. GPU DECODER — BATCH FITNESS (B=50)
    # ============================================================
    print("\n" + "="*60)
    print("3. GPU DECODER — BATCH FITNESS (B=50)")
    print("="*60)

    # Aynı 50 çözüm, bu sefer batch
    np.random.seed(0)
    seqs_50 = [np.random.permutation(n).tolist() for _ in range(50)]
    rots_50 = [np.random.uniform(0, 360, size=n).tolist() for _ in range(50)]

    # Warmup
    _ = gpu_dec.batch_fitness(seqs_50[:2], rots_50[:2])

    t0 = time.time()
    fits_50 = gpu_dec.batch_fitness(seqs_50, rots_50)
    gpu_batch_50 = time.time() - t0
    print(f'Batch(50): {gpu_batch_50:.4f}s, avg={gpu_batch_50/50:.4f}s/decode')
    print(f'Speedup vs serial GPU: {gpu_50_serial/max(gpu_batch_50, 0.001):.1f}x')
    print(f'Fitness range: [{min(fits_50):.1f}, {max(fits_50):.1f}]%')

    # 10 batch çağrı (simulates 10 iteration of pop=50)
    t0 = time.time()
    for _ in range(10):
        np.random.seed(_)
        bs = [np.random.permutation(n).tolist() for _ in range(50)]
        br = [np.random.uniform(0, 360, size=n).tolist() for _ in range(50)]
        gpu_dec.batch_fitness(bs, br)
    gpu_10x50 = time.time() - t0
    print(f'\n10 batch(50) çağrı: {gpu_10x50:.2f}s, avg batch={gpu_10x50/10:.4f}s')


    # ============================================================
    # 4. GPU DECODER — BATCH FITNESS (B=2000)
    # ============================================================
    print("\n" + "="*60)
    print("4. GPU DECODER — BATCH FITNESS (B=2000)")
    print("="*60)

    np.random.seed(42)
    seqs_2k = [np.random.permutation(n).tolist() for _ in range(2000)]
    rots_2k = [np.random.uniform(0, 360, size=n).tolist() for _ in range(2000)]

    # Warmup
    _ = gpu_dec.batch_fitness(seqs_2k[:10], rots_2k[:10])

    t0 = time.time()
    fits_2k = gpu_dec.batch_fitness(seqs_2k, rots_2k)
    gpu_batch_2k = time.time() - t0
    print(f'Batch(2000): {gpu_batch_2k:.2f}s, avg={gpu_batch_2k/2000:.6f}s/decode')
    print(f'Speedup vs serial GPU (50x40): {gpu_50_serial*40/max(gpu_batch_2k, 0.001):.1f}x')
    print(f'Fitness range: [{min(fits_2k):.1f}, {max(fits_2k):.1f}]%')
    print(f'Mean fitness: {np.mean(fits_2k):.2f}%')

    # Repeat 3x for consistent measurement
    times_2k = []
    for trial in range(3):
        np.random.seed(100 + trial)
        bs = [np.random.permutation(n).tolist() for _ in range(2000)]
        br = [np.random.uniform(0, 360, size=n).tolist() for _ in range(2000)]
        t0 = time.time()
        gpu_dec.batch_fitness(bs, br)
        times_2k.append(time.time() - t0)

    avg_2k = np.mean(times_2k)
    print(f'\n3 trial avg batch(2000): {avg_2k:.2f}s (stdev={np.std(times_2k):.3f}s)')


    # ============================================================
    # 5. KARŞILAŞTIRMA TABLOSU
    # ============================================================
    print("\n" + "="*60)
    print("5. KARŞILAŞTIRMA TABLOSU")
    print("="*60)

    # Tahminler: pop=2000, iter=50000 senaryosu
    total_evals = 2000 * 50000  # 100M

    cpu_est = cpu_avg * total_evals
    gpu_serial_est = gpu_serial_avg * total_evals
    gpu_batch50_est = (gpu_batch_50 / 50) * total_evals
    gpu_batch2k_est = avg_2k * 50000  # 50K batch çağrı, her biri 2000

    def fmt_time(secs):
        if secs < 60:
            return f"{secs:.1f}s"
        elif secs < 3600:
            return f"{secs/60:.1f} dk"
        elif secs < 86400:
            return f"{secs/3600:.1f} saat"
        else:
            return f"{secs/86400:.1f} gün"

    print(f"""
┌─────────────────────────────┬──────────────┬──────────────┬──────────────────────┐
│ Yöntem                      │ 1 eval       │ 50 eval      │ 100M eval (tahmini)  │
├─────────────────────────────┼──────────────┼──────────────┼──────────────────────┤
│ CPU BLF v3 (sıralı)        │ {cpu_avg*1000:8.2f} ms │ {cpu_50_time:8.2f} s │ {fmt_time(cpu_est):>20s} │
│ GPU tek fitness (sıralı)   │ {gpu_serial_avg*1000:8.2f} ms │ {gpu_50_serial:8.2f} s │ {fmt_time(gpu_serial_est):>20s} │
│ GPU batch (B=50)           │ {gpu_batch_50/50*1000:8.2f} ms │ {gpu_batch_50:8.2f} s │ {fmt_time(gpu_batch50_est):>20s} │
│ GPU batch (B=2000)         │ {avg_2k/2000*1000:8.2f} ms │ {avg_2k/2000*50:8.2f} s │ {fmt_time(gpu_batch2k_est):>20s} │
└─────────────────────────────┴──────────────┴──────────────┴──────────────────────┘

Speedup matrisi (vs CPU BLF sıralı):
  GPU tek:       {cpu_avg/max(gpu_serial_avg, 1e-9):.1f}x
  GPU batch(50): {cpu_avg*50/max(gpu_batch_50, 1e-6):.1f}x
  GPU batch(2K): {cpu_avg*2000/max(avg_2k, 1e-6):.1f}x

Not: device='{gpu_dec.device}' — gerçek A100 GPU'da batch(2000) çok daha hızlı olacak.
""")

    # Fitness doğrulama: CPU vs GPU aynı sonucu üretmeli
    print("--- Fitness doğrulama (CPU vs GPU) ---")
    np.random.seed(99)
    test_seq = np.random.permutation(n).tolist()
    test_rots = [0.0] * n
    f_cpu = cpu_dec.fitness(test_seq, test_rots)
    f_gpu = gpu_dec.fitness(test_seq, test_rots)
    diff = abs(f_cpu - f_gpu)
    print(f'CPU fitness: {f_cpu:.4f}%')
    print(f'GPU fitness: {f_gpu:.4f}%')
    print(f'Fark: {diff:.4f}% {"OK" if diff < 2.0 else "UYARI -- fark buyuk!"}')
