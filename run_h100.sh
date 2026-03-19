#!/bin/bash
# H100 GPU benchmark — 100M eval, 4 beden
cd /home/keremyavuz25/nesting
source venv/bin/activate

echo "=== GPU CHECK ===" | tee h100_benchmark.log
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('NO GPU')" | tee -a h100_benchmark.log

echo "" | tee -a h100_benchmark.log
echo "=== 4 BEDEN BENCHMARK — pop=2000, iter=50000 ===" | tee -a h100_benchmark.log
echo "Başlangıç: $(date)" | tee -a h100_benchmark.log
python3 -u test_4beden.py dxf_samples 2>&1 | tee -a h100_benchmark.log
echo "Bitiş: $(date)" | tee -a h100_benchmark.log
