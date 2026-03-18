#!/bin/bash
# GCP Compute Engine'e deploy et ve benchmark çalıştır.
#
# Kullanım:
#   1. gcloud CLI kurulu olmalı: https://cloud.google.com/sdk/docs/install
#   2. gcloud auth login
#   3. gcloud config set project PROJECT_ID
#   4. Bu scripti çalıştır: bash deploy-gcp.sh
#
# Tahmini maliyet: e2-standard-4 (~$0.13/saat) x ~2 saat = ~$0.26

PROJECT_ID=$(gcloud config get-value project)
ZONE="europe-west1-b"
INSTANCE="nesting-bench"
DXF_FILE="262-3061-KB-CIN.dxf"

echo "=== GCP Nesting Benchmark Deploy ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo ""

# 1. VM oluştur (CPU-only, ucuz)
echo "VM oluşturuluyor..."
gcloud compute instances create $INSTANCE \
  --zone=$ZONE \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --metadata=startup-script='#!/bin/bash
    apt-get update && apt-get install -y python3-pip python3-venv
  '

echo "30 saniye bekleniyor (VM boot)..."
sleep 30

# 2. Dosyaları kopyala
echo "Dosyalar kopyalanıyor..."
gcloud compute scp --zone=$ZONE --recurse \
  dxf_parser.py decoder.py algorithms.py export.py benchmark.py requirements.txt \
  $INSTANCE:~/nesting/

# DXF dosyasını kopyala
gcloud compute scp --zone=$ZONE \
  "../Downloads/$DXF_FILE" \
  $INSTANCE:~/nesting/test.dxf

# 3. Çalıştır
echo "Benchmark başlatılıyor..."
gcloud compute ssh $INSTANCE --zone=$ZONE --command="
  cd ~/nesting &&
  python3 -m venv venv &&
  source venv/bin/activate &&
  pip install -r requirements.txt &&
  python3 benchmark.py test.dxf 1500 2>&1 | tee benchmark.log
"

# 4. Sonuçları indir
echo "Sonuçlar indiriliyor..."
mkdir -p results
gcloud compute scp --zone=$ZONE --recurse \
  $INSTANCE:~/nesting/results/ ./results/
gcloud compute scp --zone=$ZONE \
  $INSTANCE:~/nesting/benchmark.log ./results/

# 5. VM'i sil (maliyet durur)
echo ""
read -p "VM silinsin mi? (y/n): " confirm
if [ "$confirm" = "y" ]; then
  gcloud compute instances delete $INSTANCE --zone=$ZONE --quiet
  echo "VM silindi."
else
  echo "VM çalışmaya devam ediyor: $INSTANCE"
  echo "Silmek için: gcloud compute instances delete $INSTANCE --zone=$ZONE"
fi

echo ""
echo "=== Bitti ==="
echo "Sonuçlar: results/ klasöründe"
echo "  - benchmark_report.json (karşılaştırma)"
echo "  - ssa.dxf, ga.dxf, ga_sa.dxf, ... (cutter-ready çıktılar)"
echo "  - ssa.plt, ga.plt, ga_sa.plt, ... (HPGL plotter çıktılar)"
echo "  - ssa.svg, ga.svg, ... (görsel kontrol)"
