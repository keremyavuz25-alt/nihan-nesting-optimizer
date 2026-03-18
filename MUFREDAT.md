# Nesting Optimizasyonu — Öğrenme Müfredatı
**Hedef:** 2D nesting optimizer'ı tek başına geliştirecek, iyileştirecek, debug edecek seviye.

---

## Faz 0: Temeller (1-2 hafta)
> Zaten bildiğin şeyleri hızlıca tazeleme.

### Matematik
- [ ] Koordinat geometrisi: nokta, doğru, polygon, alan hesabı
- [ ] Trigonometri: sin/cos/tan, açı dönüşümleri, radyan ↔ derece
- [ ] Matris çarpımı: 2D dönüşüm matrisleri (rotasyon, öteleme, ölçekleme)
- [ ] Shoelace formülü (polygon alanı)
- [ ] Bounding box, convex hull kavramları

**Pratik:** Kağıt üzerinde 5 parçalı bir pastalı elle yerleştir. Alanları hesapla, fire oranını bul.

### Python
- [ ] NumPy: array operasyonları, broadcasting, slicing, boolean indexing
- [ ] Shapely: Point, Polygon, intersection, union, contains, buffer
- [ ] matplotlib: polygon çizimi, imshow (bitmap görselleştirme)

**Pratik:** DXF'teki 13 parçayı matplotlib ile çiz. Her parçanın alanını Shoelace ile hesapla, Shapely ile doğrula.

---

## Faz 1: Optimizasyon Teorisi (2-3 hafta)
> Algoritmaların arkasındaki matematik.

### Kombinatorik Optimizasyon
- [ ] NP-hard nedir, neden brute force imkansız
- [ ] Arama uzayı: permütasyon (n!) x rotasyon (k^n) = toplam aday sayısı
- [ ] Fitness landscape: yerel minimum vs global minimum
- [ ] Exploitation vs exploration dengesi

**Kaynak:** "Essentials of Metaheuristics" — Sean Luke (ücretsiz PDF, 250 sayfa, çok pratik)

### Temel Algoritmalar (her birini elle takip et)
- [ ] **Hill Climbing**: en basit, yerel minimum tuzağı
- [ ] **Simulated Annealing**: sıcaklık → kabul olasılığı → Boltzmann dağılımı
  - Matematik: P(accept) = exp(-ΔE / T)
  - Soğutma şeması: T(t) = T₀ × α^t
- [ ] **Genetic Algorithm**: popülasyon → seçim → crossover → mutasyon
  - Matematik: turnuva seçimi, OX crossover (permütasyon koruyan)
  - Fitness orantılı seçim vs elit seçim
- [ ] **Particle Swarm**: hız + pozisyon güncelleme
  - Matematik: v = w·v + c₁·r₁·(pbest-x) + c₂·r₂·(gbest-x)

**Pratik:** Her algoritmayı 10 parçalık basit bin packing'de elle simüle et (spreadsheet veya kağıt). 5 iterasyon takip et.

### İleri Algoritmalar
- [ ] **Differential Evolution**: mutant vektör, F ve CR parametreleri
- [ ] **Sparrow Search**: discoverer/scrounger/scout rolleri
- [ ] **Grey Wolf**: alfa/beta/delta hiyerarşisi
- [ ] **Memetic (GA+SA)**: global arama + lokal iyileştirme neden güçlü

**Pratik:** algorithms.py'deki 8 algoritmayı oku, her birinin mutate/crossover/select adımını çiz.

---

## Faz 2: Hesaplamalı Geometri (2-3 hafta)
> Nesting'in kalbi — parçalar geometrik olarak nasıl yerleşir.

### Temel Geometri Algoritmaları
- [ ] Point-in-Polygon testi (ray casting)
- [ ] Polygon-Polygon intersection (Shapely arkasında ne var)
- [ ] Convex hull (Graham scan)
- [ ] Minkowski sum / difference (NFP'nin temeli)

### No-Fit Polygon (NFP) — KRİTİK
- [ ] NFP nedir: A parçasını B'nin etrafında kaydır, temas noktalarının izi
- [ ] NFP nasıl hesaplanır: Minkowski sum of A + mirror(B)
- [ ] Inner-Fit Rectangle (IFR): parçanın kumaş eni içinde nereye girebileceği
- [ ] NFP + IFR = tam feasible placement bölgesi

**Kaynak:** "Complete and robust no-fit polygon generation" — Burke et al. (2007)

**Pratik:** 2 basit parça (L-şekli + dikdörtgen) için NFP'yi elle çiz. Sonra Shapely ile hesapla, karşılaştır.

### Rasterize vs Vektörel
- [ ] Bitmap collision (şu anki yaklaşım): hızlı ama yaklaşık
- [ ] NFP-based placement: kesin ama hesabı ağır
- [ ] Hibrit: NFP precompute + bitmap hızlı kontrol

**Pratik:** decoder.py'deki skyline-BLF'yi oku. Darboğazı bul, NFP ile nasıl hızlanır düşün.

---

## Faz 3: Nesting'e Özel Teknikler (2-3 hafta)
> Akademik literatürden pratik teknikler.

### Placement Stratejileri
- [ ] Bottom-Left Fill (BLF) — şu anki yaklaşım
- [ ] Bottom-Left-Fill-Decreasing (BLFD) — büyük parçalar önce
- [ ] Lowest-Gap placement — en alçak boşluğa yerleştir
- [ ] NFP-based placement — feasible bölgede en iyi noktayı seç

### Encoding Stratejileri
- [ ] Sıra + rotasyon (şu anki): [permütasyon] + [açılar]
- [ ] Absolute positioning: [x₁,y₁,θ₁, x₂,y₂,θ₂, ...]
- [ ] Relative positioning: önceki parçaya göre ofset
- [ ] Her encoding'in artısı/eksisi

### Rotasyon Stratejileri
- [ ] Discrete (0/90/180/270) — şu anki
- [ ] Continuous (0-360° float) — game changer
- [ ] Grain-constrained (±5°, ±15° tolerans) — gerçek üretim
- [ ] Flip (aynalama) — simetrik parçalar için 2x seçenek

### Multi-Objective
- [ ] Sadece fire minimizasyonu değil, aynı anda:
  - Kumaş yönü (grain direction)
  - Kesim süresi (cutter yolu kısalığı)
  - Desen eşleştirme (ekose/çizgili)
- [ ] Pareto optimal: birden fazla hedefi dengeleyen çözüm kümesi

**Kaynak:** "An improved typology of cutting and packing problems" — Wäscher et al. (2007)

**Pratik:** decoder.py'ye continuous rotation ekle. 4 açı vs 360° benchmark karşılaştırması yap.

---

## Faz 4: GPU Programlama (3-4 hafta)
> Paralel hesaplama ile 50-100x hızlanma.

### PyTorch Temelleri
- [ ] Tensor: oluşturma, shape, dtype, device (CPU vs CUDA)
- [ ] Tensor operasyonları: matmul, conv2d, element-wise
- [ ] GPU'ya veri taşıma: .to('cuda'), .cpu()
- [ ] Batched operations: birden fazla input aynı anda

### GPU Nesting
- [ ] Bitmap'leri tensor'a çevir: [popülasyon, height, width]
- [ ] Paralel collision check: batch conv2d veya element-wise AND
- [ ] Paralel fitness eval: 50 bireyi aynı anda decode et
- [ ] Skyline'ı tensor'da takip etme

### Benchmark
- [ ] CPU vs GPU hız karşılaştırması (aynı algoritma)
- [ ] Optimal batch size bulma (GPU utilization)
- [ ] Memory management (büyük bitmap'ler GPU RAM'e sığıyor mu)

**Pratik:** decoder.py'nin GPU versiyonunu yaz. 50 bireyi paralel evaluate et. CPU ile karşılaştır.

---

## Faz 5: Üretim Sistemi (2-3 hafta)
> POC'dan production'a.

### Yazılım Mühendisliği
- [ ] FastAPI: REST endpoint (upload DXF → optimize → download result)
- [ ] Celery + Redis: job queue, worker pool
- [ ] Docker: containerize (API + worker + Redis)
- [ ] Testing: unit test (geometri), integration test (end-to-end)

### Monitoring
- [ ] Her job: başlangıç verimlilik, iterasyon başına iyileşme, final sonuç
- [ ] Algoritma karşılaştırma dashboard
- [ ] Firma bazlı raporlama

### Gerçek Dünya Kısıtları
- [ ] Kumaş hata/defekt haritası (DXF overlay)
- [ ] Desen eşleştirme (ekose kumaş)
- [ ] Çoklu beden yerleştirme (S+M+L aynı pastalde)
- [ ] Kumaş eni varyasyonu (kenarlar düzensiz)

**Pratik:** FastAPI endpoint yaz: POST /optimize (DXF upload) → 202 Accepted → GET /result/{id}

---

## Okuma Listesi

### Kitaplar
1. **"Essentials of Metaheuristics"** — Sean Luke (ÜCRETSİZ)
   cs.gmu.edu/~sean/book/metaheuristics/
   En pratik metaheuristic kitabı. Pseudocode + açıklama.

2. **"Cutting and Packing in Production and Distribution"** — Dyckhoff & Finke
   Nesting/packing teorisi.

3. **"Computational Geometry: Algorithms and Applications"** — de Berg et al.
   NFP, Minkowski sum, intersection algoritmaları.

### Makaleler
1. Burke et al. (2007) — "Complete and robust no-fit polygon generation"
   NFP'nin standart referansı.

2. Bennell & Oliveira (2008) — "The geometry of nesting problems"
   Nesting geometrisinin kapsamlı survey'ı.

3. Elkeran (2013) — "A new approach for sheet nesting problem using guided cuckoo search"
   Metaheuristic + nesting kombinasyonu.

4. Xue & Zhu (2023) — "Sparrow search algorithm"
   SSA orijinal makale.

### Video
1. **3Blue1Brown** — Linear Algebra serisi (matris dönüşümleri)
2. **Sebastian Lague** — Computational Geometry (YouTube)
3. **Reducible** — Simulated Annealing açıklaması

---

## Haftalık Plan

| Hafta | Konu | Çıktı |
|-------|-------|--------|
| 1 | Faz 0: NumPy + Shapely + DXF çizim | 13 parça matplotlib plot |
| 2 | Faz 1: SA + GA elle simülasyon | Kağıt üzerinde 5 iterasyon takibi |
| 3 | Faz 1: 8 algoritmayı oku + anla | Her algoritmanın akış şeması |
| 4 | Faz 2: Point-in-polygon, intersection | Kendi collision checker'ın |
| 5 | Faz 2: NFP hesaplama | 2 parça NFP'si elle + kodla |
| 6 | Faz 3: Continuous rotation | 4 açı vs 360° benchmark |
| 7 | Faz 3: NFP-based decoder | Yeni decoder, verimlilik karşılaştırma |
| 8 | Faz 4: PyTorch tensor basics | GPU'da bitmap collision |
| 9 | Faz 4: Paralel fitness eval | 50x hızlanma benchmark |
| 10 | Faz 5: FastAPI + Docker | Çalışan API endpoint |
| 11 | Faz 5: Gerçek dünya kısıtları | Grain direction, defekt |
| 12 | Final: Lectra karşılaştırma | Gerçek pastal A/B testi |

---

## Önemli Not
Bu müfredat lineer değil — Faz 0-1'i bitirince geri kalanını proje ihtiyacına göre paralel ilerletebilirsin. En kritik sıra:
1. Optimizasyon teorisi (Faz 1) — algoritmaları anlamadan ilerleyemezsin
2. NFP (Faz 2) — verimlilik artışının %80'i buradan gelir
3. GPU (Faz 4) — hız artışının %90'ı buradan gelir
