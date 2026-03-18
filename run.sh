#!/bin/bash
cd /home/keremyavuz25/nesting
source venv/bin/activate
python3 -u benchmark.py test.dxf 1500 > benchmark.log 2>&1
echo "DONE" >> benchmark.log
