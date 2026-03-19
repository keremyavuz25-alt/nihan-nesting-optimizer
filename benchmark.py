#!/usr/bin/env python3
"""Nesting Algoritma Benchmark — 8 algoritma PARALEL, BLF veya NFP decoder."""
import sys
import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from dxf_parser import load_dxf
from export import export_dxf, export_plt, export_svg


def run_single_algorithm(args):
    """Tek algoritmayı çalıştır (paralel worker'da)."""
    alg_name, dxf_path, bin_width, pop_size, max_iter, seed, decoder_type = args

    from dxf_parser import load_dxf
    pieces = load_dxf(dxf_path)
    n = len(pieces)

    # Decoder seç
    if decoder_type == "nfp":
        from nfp_decoder import NFPDecoder
        decoder = NFPDecoder(pieces, bin_width=bin_width)
    else:
        from decoder import BLFDecoder
        decoder = BLFDecoder(pieces, bin_width=bin_width, resolution=3.0)

    np.random.seed(seed)

    from algorithms import (
        sparrow_search, genetic_algorithm, ga_sa_hybrid,
        simulated_annealing, differential_evolution,
        particle_swarm, grey_wolf, tabu_search,
    )

    sa_total = pop_size * max_iter

    alg_map = {
        "SSA": lambda: sparrow_search(decoder.fitness, n, pop_size, max_iter, verbose=False),
        "GA": lambda: genetic_algorithm(decoder.fitness, n, pop_size, max_iter, verbose=False),
        "GA+SA": lambda: ga_sa_hybrid(decoder.fitness, n, pop_size, max_iter, verbose=False),
        "SA": lambda: simulated_annealing(decoder.fitness, n, sa_total, verbose=False),
        "DE": lambda: differential_evolution(decoder.fitness, n, pop_size, max_iter, verbose=False),
        "PSO": lambda: particle_swarm(decoder.fitness, n, pop_size, max_iter, verbose=False),
        "GWO": lambda: grey_wolf(decoder.fitness, n, pop_size, max_iter, verbose=False),
        "Tabu": lambda: tabu_search(decoder.fitness, n, sa_total, verbose=False),
    }

    result = alg_map[alg_name]()
    return alg_name, result


def run_benchmark(dxf_path: str, bin_width: float = 1500.0,
                  pop_size: int = 50, max_iter: int = 5000,
                  n_runs: int = 1, output_dir: str = "results",
                  decoder_type: str = "blf"):
    """Tüm algoritmaları PARALEL çalıştır."""

    print(f"{'='*60}")
    print(f"NESTING BENCHMARK (PARALEL)")
    print(f"{'='*60}")
    print(f"DXF: {dxf_path}")
    print(f"Decoder: {decoder_type.upper()}")
    print(f"Kumaş eni: {bin_width} mm")
    print(f"Popülasyon: {pop_size}, İterasyon: {max_iter}, Tekrar: {n_runs}")
    print(f"CPU sayısı: {cpu_count()}")
    print()

    # 1. DXF parse
    print("DXF yükleniyor...")
    pieces = load_dxf(dxf_path)
    total_area = sum(p["area"] for p in pieces)
    print(f"  {len(pieces)} parça, toplam alan: {total_area/100:.0f} cm²")
    for p in pieces:
        print(f"    #{p['id']}: {p['width']:.0f}x{p['height']:.0f}mm, {p['area']/100:.0f}cm²")
    print()

    # 2. Tüm işleri hazırla
    alg_names = ["SSA", "GA", "GA+SA", "SA", "DE", "PSO", "GWO", "Tabu"]
    jobs = []
    for alg_name in alg_names:
        for r in range(n_runs):
            jobs.append((alg_name, dxf_path, bin_width, pop_size, max_iter, 42 + r, decoder_type))

    print(f"Toplam {len(jobs)} iş, {min(len(jobs), cpu_count())} paralel worker")
    print(f"Başlatılıyor...\n")
    sys.stdout.flush()

    # 3. Paralel çalıştır
    os.makedirs(output_dir, exist_ok=True)
    results_by_alg = {name: [] for name in alg_names}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=min(len(jobs), cpu_count())) as executor:
        futures = {executor.submit(run_single_algorithm, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            try:
                alg_name, result = future.result()
                results_by_alg[alg_name].append(result)
                print(f"  {alg_name}: {result['best_fitness']:.2f}% ({result['time']:.1f}s)")
                sys.stdout.flush()
            except Exception as e:
                print(f"  {job[0]}: HATA — {e}")
                sys.stdout.flush()

    t_total = time.time() - t_start
    print(f"\nToplam süre: {t_total:.0f}s ({t_total/60:.1f}dk)")

    # 4. Sonuçları derle
    all_results = []

    # Export için decoder oluştur
    if decoder_type == "nfp":
        from nfp_decoder import NFPDecoder
        decoder = NFPDecoder(pieces, bin_width=bin_width)
    else:
        from decoder import BLFDecoder
        decoder = BLFDecoder(pieces, bin_width=bin_width, resolution=3.0)

    for alg_name in alg_names:
        runs = results_by_alg[alg_name]
        if not runs:
            continue

        best_run = max(runs, key=lambda x: x["best_fitness"])
        fits = [r["best_fitness"] for r in runs]
        times = [r["time"] for r in runs]

        summary = {
            "algorithm": best_run["name"],
            "best": max(fits),
            "worst": min(fits),
            "mean": float(np.mean(fits)),
            "std": float(np.std(fits)),
            "avg_time": float(np.mean(times)),
            "runs": len(runs),
        }
        all_results.append(summary)

        # En iyi çözümü export et
        best_seq, best_rots = best_run["best_solution"]
        layout = decoder.decode(best_seq, best_rots)

        prefix = os.path.join(output_dir, alg_name.lower().replace("+", "_"))
        export_dxf(layout["placements"], pieces, bin_width, layout["used_length"], f"{prefix}.dxf")
        export_plt(layout["placements"], pieces, bin_width, layout["used_length"], f"{prefix}.plt")
        export_svg(layout["placements"], pieces, bin_width, layout["used_length"], f"{prefix}.svg")

    # 5. Karşılaştırma tablosu
    print(f"\n\n{'='*70}")
    print(f"DECODER: {decoder_type.upper()}")
    print(f"{'ALGORİTMA':<25} {'EN İYİ':>8} {'ORT':>8} {'STD':>6} {'SÜRE':>8}")
    print(f"{'='*70}")

    all_results.sort(key=lambda x: x["best"], reverse=True)
    for r in all_results:
        print(f"{r['algorithm']:<25} {r['best']:>7.2f}% {r['mean']:>7.2f}% {r['std']:>5.2f} {r['avg_time']:>7.1f}s")

    print(f"{'='*70}")
    if all_results:
        winner = all_results[0]
        print(f"\nKAZANAN: {winner['algorithm']} — {winner['best']:.2f}%")

    print(f"Toplam wall-clock: {t_total:.0f}s ({t_total/60:.1f}dk)")

    # JSON rapor
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dxf": os.path.basename(dxf_path),
            "decoder": decoder_type,
            "pieces": len(pieces),
            "total_area_cm2": total_area / 100,
            "bin_width_mm": bin_width,
            "pop_size": pop_size,
            "max_iter": max_iter,
            "n_runs": n_runs,
            "wall_clock_seconds": t_total,
            "cpu_count": cpu_count(),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nRapor: {report_path}")

    return all_results


if __name__ == "__main__":
    dxf = sys.argv[1] if len(sys.argv) > 1 else "test.dxf"
    width = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0
    dec = sys.argv[3] if len(sys.argv) > 3 else "blf"

    run_benchmark(
        dxf_path=dxf,
        bin_width=width,
        pop_size=50,
        max_iter=5000,
        n_runs=1,
        output_dir=f"results_{dec}",
        decoder_type=dec,
    )
