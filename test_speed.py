"""Raster BLF decoder hız testi."""
from dxf_parser import load_dxf
from decoder import BLFDecoder
import time, numpy as np

pieces = load_dxf('test.dxf')
print(f'{len(pieces)} pieces loaded')

# Resolution=5mm (hızlı)
print("Bitmap cache oluşturuluyor (res=5mm)...")
t0 = time.time()
decoder = BLFDecoder(pieces, bin_width=1500, resolution=5.0)
t1 = time.time()
print(f"Cache süresi: {t1-t0:.2f}s")

np.random.seed(42)
seq = np.random.permutation(len(pieces)).tolist()
rots = [0]*len(pieces)

t0 = time.time()
result = decoder.decode(seq, rots)
t1 = time.time()
print(f'Res=5mm, single decode: {t1-t0:.3f}s, util={result["utilization"]:.2f}%, length={result["used_length"]:.0f}mm')

# 10 decode
t0 = time.time()
for i in range(10):
    s = np.random.permutation(len(pieces)).tolist()
    r = np.random.choice([0, 90, 180, 270], size=len(pieces)).tolist()
    decoder.decode(s, r)
t1 = time.time()
avg = (t1-t0)/10
print(f'10 decode: {t1-t0:.2f}s, avg={avg:.3f}s/decode')
print(f'Tahmini 100K decode: {avg*100000/3600:.1f} saat')
print(f'Tahmini 1M decode: {avg*1000000/3600:.1f} saat')
