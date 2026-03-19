"""Decoder v2 hız testi — continuous rotation + LRU cache."""
from dxf_parser import load_dxf
from decoder import BLFDecoder
import time, numpy as np

pieces = load_dxf('test.dxf')
print(f'{len(pieces)} pieces loaded')

# Resolution=3mm (daha hassas)
print("\nCache oluşturuluyor (res=3mm, 4 cardinal açı)...")
t0 = time.time()
decoder = BLFDecoder(pieces, bin_width=1500, resolution=3.0)
t1 = time.time()
print(f"Cache süresi: {t1-t0:.2f}s")

# Test 1: Cardinal açı (cache hit)
np.random.seed(42)
seq = np.random.permutation(len(pieces)).tolist()
rots = [0.0]*len(pieces)
t0 = time.time()
result = decoder.decode(seq, rots)
t1 = time.time()
print(f'\nCardinal (cache hit): {t1-t0:.4f}s, util={result["utilization"]:.2f}%, placed={result["n_placed"]}/{len(pieces)}')

# Test 2: Continuous açı (cache miss → build)
rots_cont = np.random.uniform(0, 360, size=len(pieces)).tolist()
t0 = time.time()
result2 = decoder.decode(seq, rots_cont)
t1 = time.time()
print(f'Continuous (cache miss): {t1-t0:.4f}s, util={result2["utilization"]:.2f}%, placed={result2["n_placed"]}/{len(pieces)}')

# Test 3: Continuous açı (cache hit — same angles)
t0 = time.time()
result3 = decoder.decode(seq, rots_cont)
t1 = time.time()
print(f'Continuous (cache hit): {t1-t0:.4f}s, util={result3["utilization"]:.2f}%')

# Test 4: 50 decode benchmark (simulates 1 iteration of pop=50)
print(f'\n50 decode benchmark (continuous rotation)...')
t0 = time.time()
for i in range(50):
    s = np.random.permutation(len(pieces)).tolist()
    r = np.random.uniform(0, 360, size=len(pieces)).tolist()
    decoder.decode(s, r)
t1 = time.time()
avg = (t1-t0)/50
print(f'50 decode: {t1-t0:.2f}s, avg={avg:.4f}s/decode')
print(f'Cache size: {len(decoder._cache)} entries')

# Tahminler
print(f'\n--- Tahminler ---')
print(f'500 iter x 50 pop = 25K decode: {avg*25000/60:.1f} dk')
print(f'2000 iter x 50 pop = 100K decode: {avg*100000/60:.1f} dk')
print(f'5000 iter x 50 pop = 250K decode: {avg*250000/3600:.1f} saat')

# Fitness testi (penalty)
print(f'\n--- Fitness (penalty) testi ---')
f1 = decoder.fitness(seq, rots)
print(f'Cardinal: fitness={f1:.2f}% (util={result["utilization"]:.2f}%, placed={result["n_placed"]})')
f2 = decoder.fitness(seq, rots_cont)
print(f'Continuous: fitness={f2:.2f}% (util={result2["utilization"]:.2f}%, placed={result2["n_placed"]})')
