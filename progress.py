"""Canlı progress takibi — benchmark çalışırken en iyi sonucu gösterir.

Kullanım (ayrı terminal/hücre):
  python progress.py

Veya Colab'da:
  !python progress.py &
  !python test_4beden.py
"""
import time, os, json, sys

PROGRESS_FILE = "progress.json"

def write_progress(model: str, iteration: int, max_iter: int,
                   best_fitness: float, elapsed: float):
    """Algoritma içinden çağrılır — mevcut durumu dosyaya yaz."""
    data = {
        "model": model,
        "iteration": iteration,
        "max_iter": max_iter,
        "best_fitness": best_fitness,
        "elapsed": elapsed,
        "timestamp": time.time(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)


def watch():
    """Progress dosyasını izle ve canlı göster."""
    last_ts = 0
    print("Progress izleniyor... (Ctrl+C ile dur)")
    print()

    while True:
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE) as f:
                    data = json.load(f)

                ts = data.get("timestamp", 0)
                if ts > last_ts:
                    last_ts = ts
                    pct = data["iteration"] / max(data["max_iter"], 1) * 100
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)

                    sys.stdout.write(
                        f'\r  {data["model"]:<25s} |{bar}| '
                        f'{pct:>5.1f}% iter={data["iteration"]}/{data["max_iter"]} '
                        f'best={data["best_fitness"]:.1f}% '
                        f'({data["elapsed"]:.0f}s)  '
                    )
                    sys.stdout.flush()

            time.sleep(0.5)
        except (json.JSONDecodeError, KeyError):
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nDurdu.")
            break


if __name__ == "__main__":
    watch()
