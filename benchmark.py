#!/usr/bin/env python3
"""Nesting Algoritma Benchmark — 8 algoritma, DXF girdi, DXF/PLT/SVG çıktı."""
import sys
import os
import json
import time
import numpy as np

from dxf_parser import load_dxf
from decoder import BLFDecoder
from algorithms import (
    sparrow_search,
    genetic_algorithm,
    ga_sa_hybrid,
    simulated_annealing,
    differential_evolution,
    particle_swarm,
    grey_wolf,
    tabu_search,
)
from export import export_dxf, export_plt, export_svg


def run_benchmark(dxf_path: str, bin_width: float = 1500.0,
                  pop_size: int = 50, max_iter: int = 2000,
                  n_runs: int = 3, output_dir: str = "results"):
    """Tüm algoritmaları çalıştır ve karşılaştır."""

    print(f"{'='*60}")
    print(f"NESTING BENCHMARK")
    print(f"{'='*60}")
    print(f"DXF: {dxf_path}")
    print(f"Kumaş eni: {bin_width} mm")
    print(f"Popülasyon: {pop_size}, İterasyon: {max_iter}, Tekrar: {n_runs}")
    print()

    # 1. DXF parse
    print("DXF yükleniyor...")
    pieces = load_dxf(dxf_path)
    total_area = sum(p["area"] for p in pieces)
    print(f"  {len(pieces)} parça, toplam alan: {total_area/100:.0f} cm²")
    for p in pieces:
        print(f"    #{p['id']}: {p['width']:.0f}x{p['height']:.0f}mm, {p['area']/100:.0f}cm²")
    print()

    # 2. Decoder oluştur
    decoder = BLFDecoder(pieces, bin_width=bin_width, resolution=5.0)

    # 3. Algoritmaları tanımla
    sa_total = pop_size * max_iter  # SA ve Tabu için eşit eval sayısı

    algorithms = [
        ("SSA", lambda: sparrow_search(decoder.fitness, len(pieces), pop_size, max_iter, verbose=True)),
        ("GA", lambda: genetic_algorithm(decoder.fitness, len(pieces), pop_size, max_iter, verbose=True)),
        ("GA+SA", lambda: ga_sa_hybrid(decoder.fitness, len(pieces), pop_size, max_iter, verbose=True)),
        ("SA", lambda: simulated_annealing(decoder.fitness, len(pieces), sa_total, verbose=True)),
        ("DE", lambda: differential_evolution(decoder.fitness, len(pieces), pop_size, max_iter, verbose=True)),
        ("PSO", lambda: particle_swarm(decoder.fitness, len(pieces), pop_size, max_iter, verbose=True)),
        ("GWO", lambda: grey_wolf(decoder.fitness, len(pieces), pop_size, max_iter, verbose=True)),
        ("Tabu", lambda: tabu_search(decoder.fitness, len(pieces), sa_total, verbose=True)),
    ]

    # 4. Benchmark çalıştır
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for alg_name, alg_fn in algorithms:
        print(f"\n{'─'*50}")
        print(f"▶ {alg_name} ({n_runs} tekrar)")
        print(f"{'─'*50}")

        runs = []
        for r in range(n_runs):
            print(f"\n  Run {r+1}/{n_runs}:")
            np.random.seed(42 + r)
            result = alg_fn()
            runs.append(result)
            print(f"  → {result['best_fitness']:.2f}% ({result['time']:.1f}s)")

        # En iyi run
        best_run = max(runs, key=lambda x: x["best_fitness"])

        # İstatistikler
        fits = [r["best_fitness"] for r in runs]
        times = [r["time"] for r in runs]

        summary = {
            "algorithm": best_run["name"],
            "best": max(fits),
            "worst": min(fits),
            "mean": np.mean(fits),
            "std": np.std(fits),
            "avg_time": np.mean(times),
            "runs": len(runs),
        }
        all_results.append(summary)

        print(f"\n  📊 {alg_name}: best={summary['best']:.2f}% mean={summary['mean']:.2f}% "
              f"std={summary['std']:.2f} time={summary['avg_time']:.1f}s")

        # En iyi çözümü decode et ve dışa aktar
        best_seq, best_rots = best_run["best_solution"]
        layout = decoder.decode(best_seq, best_rots)

        prefix = os.path.join(output_dir, alg_name.lower().replace("+", "_"))
        export_dxf(layout["placements"], pieces, bin_width, layout["used_length"], f"{prefix}.dxf")
        export_plt(layout["placements"], pieces, bin_width, layout["used_length"], f"{prefix}.plt")
        export_svg(layout["placements"], pieces, bin_width, layout["used_length"], f"{prefix}.svg")
        print(f"  💾 {prefix}.dxf / .plt / .svg")

    # 5. Karşılaştırma tablosu
    print(f"\n\n{'='*70}")
    print(f"{'ALGORİTMA':<25} {'EN İYİ':>8} {'ORT':>8} {'STD':>6} {'SÜRE':>8}")
    print(f"{'='*70}")

    all_results.sort(key=lambda x: x["best"], reverse=True)
    for r in all_results:
        print(f"{r['algorithm']:<25} {r['best']:>7.2f}% {r['mean']:>7.2f}% {r['std']:>5.2f} {r['avg_time']:>7.1f}s")

    print(f"{'='*70}")
    winner = all_results[0]
    print(f"\n🏆 KAZANAN: {winner['algorithm']} — {winner['best']:.2f}%")

    # JSON rapor
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dxf": os.path.basename(dxf_path),
            "pieces": len(pieces),
            "total_area_cm2": total_area / 100,
            "bin_width_mm": bin_width,
            "pop_size": pop_size,
            "max_iter": max_iter,
            "n_runs": n_runs,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📋 Rapor: {report_path}")

    return all_results


if __name__ == "__main__":
    dxf = sys.argv[1] if len(sys.argv) > 1 else "test.dxf"
    width = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0

    run_benchmark(
        dxf_path=dxf,
        bin_width=width,
        pop_size=50,
        max_iter=2000,
        n_runs=3,
        output_dir="results",
    )
