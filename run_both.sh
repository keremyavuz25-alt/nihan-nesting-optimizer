#!/bin/bash
cd /home/keremyavuz25/nesting
source venv/bin/activate

echo "=== BLF BENCHMARK BAŞLIYOR ===" | tee -a benchmark_all.log
echo "Başlangıç: $(date)" | tee -a benchmark_all.log
python3 -u benchmark.py test.dxf 1500 blf 2>&1 | tee -a benchmark_all.log

echo "" | tee -a benchmark_all.log
echo "=== NFP BENCHMARK BAŞLIYOR ===" | tee -a benchmark_all.log
echo "Başlangıç: $(date)" | tee -a benchmark_all.log
python3 -u benchmark.py test.dxf 1500 nfp 2>&1 | tee -a benchmark_all.log

echo "" | tee -a benchmark_all.log
echo "=== TAMAMLANDI ===" | tee -a benchmark_all.log
echo "Bitiş: $(date)" | tee -a benchmark_all.log
