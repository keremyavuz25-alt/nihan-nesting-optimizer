"""8 optimizasyon algoritması — nesting benchmark için."""
import numpy as np
from copy import deepcopy
import time


# ============================================================
# ORTAK YARDIMCILAR
# ============================================================

def random_solution(n: int) -> tuple:
    """Rastgele sıra + rotasyon üret."""
    seq = np.random.permutation(n).tolist()
    rots = np.random.choice([0, 90, 180, 270], size=n).tolist()
    return seq, rots


def mutate_sequence(seq: list, strength: int = 1) -> list:
    """Sırada swap mutasyonu."""
    s = seq.copy()
    for _ in range(strength):
        i, j = np.random.choice(len(s), 2, replace=False)
        s[i], s[j] = s[j], s[i]
    return s


def mutate_rotation(rots: list, strength: int = 1) -> list:
    """Rotasyonda rastgele değişiklik."""
    r = rots.copy()
    for _ in range(strength):
        i = np.random.randint(len(r))
        r[i] = np.random.choice([0, 90, 180, 270])
    return r


def crossover_order(p1: list, p2: list) -> list:
    """Order Crossover (OX) — permütasyon koruyan."""
    n = len(p1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child


def crossover_rotation(r1: list, r2: list) -> list:
    """Tek nokta crossover rotasyonlar için."""
    point = np.random.randint(len(r1))
    return r1[:point] + r2[point:]


# ============================================================
# 1. SPARROW SEARCH ALGORITHM (SSA)
# ============================================================

def sparrow_search(fitness_fn, n_pieces: int, pop_size: int = 50,
                   max_iter: int = 5000, verbose: bool = False) -> dict:
    """Sparrow Search Algorithm for nesting."""
    PD = 0.2  # discoverer oranı
    SD = 0.1  # scrounger-den sınırı
    n_disc = max(1, int(pop_size * PD))

    # Başlangıç popülasyonu
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    best_idx = np.argmax(fits)
    best_sol = deepcopy(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit]

    t0 = time.time()
    for it in range(max_iter):
        R2 = np.random.random()  # alarm değeri

        # Discoverer'lar (en iyi %20)
        sorted_idx = np.argsort(fits)[::-1]
        for i in range(n_disc):
            idx = sorted_idx[i]
            seq, rots = pop[idx]
            if R2 < 0.8:  # güvenli — keşif
                new_seq = mutate_sequence(seq, strength=max(1, int(3 * (1 - it/max_iter))))
                new_rots = mutate_rotation(rots, strength=max(1, int(2 * (1 - it/max_iter))))
            else:  # tehlike — sömürü
                new_seq = mutate_sequence(seq, strength=1)
                new_rots = mutate_rotation(rots, strength=1)
            new_fit = fitness_fn(new_seq, new_rots)
            if new_fit > fits[idx]:
                pop[idx] = (new_seq, new_rots)
                fits[idx] = new_fit

        # Scrounger'lar (geri kalan)
        worst_idx = sorted_idx[-1]
        for i in range(n_disc, pop_size):
            idx = sorted_idx[i]
            seq, rots = pop[idx]
            if fits[idx] > fits[worst_idx]:
                # En iyiye doğru hareket
                new_seq = crossover_order(seq, best_sol[0])
                new_rots = crossover_rotation(rots, best_sol[1])
            else:
                # Rastgele keşif
                new_seq = mutate_sequence(seq, strength=3)
                new_rots = mutate_rotation(rots, strength=3)
            new_fit = fitness_fn(new_seq, new_rots)
            if new_fit > fits[idx]:
                pop[idx] = (new_seq, new_rots)
                fits[idx] = new_fit

        # Scout (en kötüleri yenile)
        n_scout = max(1, int(pop_size * SD))
        worst_indices = sorted_idx[-n_scout:]
        for idx in worst_indices:
            if np.random.random() < 0.5:
                pop[idx] = random_solution(n_pieces)
                fits[idx] = fitness_fn(*pop[idx])

        # Global best güncelle
        cur_best = np.argmax(fits)
        if fits[cur_best] > best_fit:
            best_fit = fits[cur_best]
            best_sol = deepcopy(pop[cur_best])

        history.append(best_fit)

        if verbose and it % 500 == 0:
            print(f"  SSA iter {it}: {best_fit:.2f}%")

    return {
        "name": "SSA (Sparrow Search)",
        "best_fitness": best_fit,
        "best_solution": best_sol,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 2. GENETIC ALGORITHM (GA)
# ============================================================

def genetic_algorithm(fitness_fn, n_pieces: int, pop_size: int = 50,
                      max_iter: int = 5000, verbose: bool = False) -> dict:
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    best_idx = np.argmax(fits)
    best_sol = deepcopy(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit]

    t0 = time.time()
    for it in range(max_iter):
        # Tournament selection + crossover + mutation
        new_pop = [deepcopy(best_sol)]  # elitism
        new_fits = [best_fit]

        while len(new_pop) < pop_size:
            # Tournament
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            p1 = pop[t1] if fits[t1] > fits[t2] else pop[t2]
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            p2 = pop[t1] if fits[t1] > fits[t2] else pop[t2]

            # Crossover
            child_seq = crossover_order(p1[0], p2[0])
            child_rots = crossover_rotation(p1[1], p2[1])

            # Mutation
            if np.random.random() < 0.3:
                child_seq = mutate_sequence(child_seq)
            if np.random.random() < 0.2:
                child_rots = mutate_rotation(child_rots)

            child_fit = fitness_fn(child_seq, child_rots)
            new_pop.append((child_seq, child_rots))
            new_fits.append(child_fit)

        pop = new_pop[:pop_size]
        fits = new_fits[:pop_size]

        cur_best = np.argmax(fits)
        if fits[cur_best] > best_fit:
            best_fit = fits[cur_best]
            best_sol = deepcopy(pop[cur_best])

        history.append(best_fit)

        if verbose and it % 500 == 0:
            print(f"  GA iter {it}: {best_fit:.2f}%")

    return {
        "name": "GA (Genetic Algorithm)",
        "best_fitness": best_fit,
        "best_solution": best_sol,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 3. GA + SA (Hibrit Memetic)
# ============================================================

def ga_sa_hybrid(fitness_fn, n_pieces: int, pop_size: int = 50,
                 max_iter: int = 5000, sa_iters: int = 50, verbose: bool = False) -> dict:
    """GA global arama + SA lokal iyileştirme (her 100 iterasyonda en iyiye)."""
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    best_idx = np.argmax(fits)
    best_sol = deepcopy(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit]

    t0 = time.time()
    for it in range(max_iter):
        # --- GA adımı ---
        new_pop = [deepcopy(best_sol)]
        new_fits = [best_fit]

        while len(new_pop) < pop_size:
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            p1 = pop[t1] if fits[t1] > fits[t2] else pop[t2]
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            p2 = pop[t1] if fits[t1] > fits[t2] else pop[t2]

            child_seq = crossover_order(p1[0], p2[0])
            child_rots = crossover_rotation(p1[1], p2[1])

            if np.random.random() < 0.3:
                child_seq = mutate_sequence(child_seq)
            if np.random.random() < 0.2:
                child_rots = mutate_rotation(child_rots)

            new_pop.append((child_seq, child_rots))
            new_fits.append(fitness_fn(child_seq, child_rots))

        pop = new_pop[:pop_size]
        fits = new_fits[:pop_size]

        # --- SA adımı (her 100 iterasyonda en iyiye) ---
        if it % 100 == 0 and it > 0:
            cur_seq, cur_rots = deepcopy(best_sol)
            cur_fit = best_fit
            temp = 10.0

            for sa_it in range(sa_iters):
                new_seq = mutate_sequence(cur_seq, strength=1)
                new_rots = mutate_rotation(cur_rots, strength=1)
                new_fit = fitness_fn(new_seq, new_rots)

                delta = new_fit - cur_fit
                if delta > 0 or np.random.random() < np.exp(delta / max(temp, 0.01)):
                    cur_seq, cur_rots = new_seq, new_rots
                    cur_fit = new_fit

                temp *= 0.95

            if cur_fit > best_fit:
                best_fit = cur_fit
                best_sol = (cur_seq, cur_rots)
                # En iyiyi popülasyona geri koy
                worst = np.argmin(fits)
                pop[worst] = deepcopy(best_sol)
                fits[worst] = best_fit

        cur_best = np.argmax(fits)
        if fits[cur_best] > best_fit:
            best_fit = fits[cur_best]
            best_sol = deepcopy(pop[cur_best])

        history.append(best_fit)

        if verbose and it % 500 == 0:
            print(f"  GA+SA iter {it}: {best_fit:.2f}%")

    return {
        "name": "GA+SA (Hibrit)",
        "best_fitness": best_fit,
        "best_solution": best_sol,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 4. SIMULATED ANNEALING (SA) — baseline
# ============================================================

def simulated_annealing(fitness_fn, n_pieces: int, max_iter: int = 250000,
                        verbose: bool = False) -> dict:
    """SA tek çözüm, toplam eval sayısı diğerleriyle eşit (pop_size * max_iter)."""
    seq, rots = random_solution(n_pieces)
    cur_fit = fitness_fn(seq, rots)
    best_sol = (seq.copy(), rots.copy())
    best_fit = cur_fit
    history = [best_fit]

    temp = 50.0
    cooling = 0.99995

    t0 = time.time()
    for it in range(max_iter):
        new_seq = mutate_sequence(seq, strength=1 + int(temp > 10))
        new_rots = mutate_rotation(rots, strength=1)
        new_fit = fitness_fn(new_seq, new_rots)

        delta = new_fit - cur_fit
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 0.001)):
            seq, rots = new_seq, new_rots
            cur_fit = new_fit

        if cur_fit > best_fit:
            best_fit = cur_fit
            best_sol = (seq.copy(), rots.copy())

        temp *= cooling
        history.append(best_fit)

        if verbose and it % 25000 == 0:
            print(f"  SA iter {it}: {best_fit:.2f}% (T={temp:.2f})")

    return {
        "name": "SA (Simulated Annealing)",
        "best_fitness": best_fit,
        "best_solution": best_sol,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 5. DIFFERENTIAL EVOLUTION (DE)
# ============================================================

def differential_evolution(fitness_fn, n_pieces: int, pop_size: int = 50,
                           max_iter: int = 5000, verbose: bool = False) -> dict:
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    best_idx = np.argmax(fits)
    best_sol = deepcopy(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit]

    t0 = time.time()
    F = 0.8  # scale factor
    CR = 0.7  # crossover rate

    for it in range(max_iter):
        for i in range(pop_size):
            # 3 rastgele farklı birey seç
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            seq_i, rots_i = pop[i]
            seq_a, rots_a = pop[a]
            seq_b, rots_b = pop[b]
            seq_c, rots_c = pop[c]

            # Mutant vektör: sıra için crossover, rotasyon için differential
            trial_seq = seq_i.copy()
            trial_rots = rots_i.copy()

            for j in range(n_pieces):
                if np.random.random() < CR:
                    trial_seq = crossover_order(seq_a, seq_b) if np.random.random() < F else trial_seq
                    trial_rots[j] = rots_a[j] if np.random.random() < F else rots_c[j]

            trial_fit = fitness_fn(trial_seq, trial_rots)
            if trial_fit > fits[i]:
                pop[i] = (trial_seq, trial_rots)
                fits[i] = trial_fit

        cur_best = np.argmax(fits)
        if fits[cur_best] > best_fit:
            best_fit = fits[cur_best]
            best_sol = deepcopy(pop[cur_best])

        history.append(best_fit)

        if verbose and it % 500 == 0:
            print(f"  DE iter {it}: {best_fit:.2f}%")

    return {
        "name": "DE (Differential Evolution)",
        "best_fitness": best_fit,
        "best_solution": best_sol,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 6. PARTICLE SWARM OPTIMIZATION (PSO)
# ============================================================

def particle_swarm(fitness_fn, n_pieces: int, pop_size: int = 50,
                   max_iter: int = 5000, verbose: bool = False) -> dict:
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]
    pbest = deepcopy(pop)
    pbest_fits = fits.copy()

    best_idx = np.argmax(fits)
    gbest = deepcopy(pop[best_idx])
    gbest_fit = fits[best_idx]
    history = [gbest_fit]

    t0 = time.time()
    for it in range(max_iter):
        w = 0.9 - 0.5 * (it / max_iter)  # inertia azalır

        for i in range(pop_size):
            seq, rots = pop[i]

            # Velocity = mutation strength
            # Cognitive: pbest'e doğru
            if np.random.random() < 0.5:
                seq = crossover_order(seq, pbest[i][0])
                rots = crossover_rotation(rots, pbest[i][1])

            # Social: gbest'e doğru
            if np.random.random() < 0.5:
                seq = crossover_order(seq, gbest[0])
                rots = crossover_rotation(rots, gbest[1])

            # Inertia: rastgele mutasyon
            if np.random.random() < w:
                seq = mutate_sequence(seq)
                rots = mutate_rotation(rots)

            new_fit = fitness_fn(seq, rots)
            pop[i] = (seq, rots)
            fits[i] = new_fit

            if new_fit > pbest_fits[i]:
                pbest[i] = deepcopy(pop[i])
                pbest_fits[i] = new_fit

        cur_best = np.argmax(fits)
        if fits[cur_best] > gbest_fit:
            gbest_fit = fits[cur_best]
            gbest = deepcopy(pop[cur_best])

        history.append(gbest_fit)

        if verbose and it % 500 == 0:
            print(f"  PSO iter {it}: {gbest_fit:.2f}%")

    return {
        "name": "PSO (Particle Swarm)",
        "best_fitness": gbest_fit,
        "best_solution": gbest,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 7. GREY WOLF OPTIMIZER (GWO)
# ============================================================

def grey_wolf(fitness_fn, n_pieces: int, pop_size: int = 50,
              max_iter: int = 5000, verbose: bool = False) -> dict:
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    sorted_idx = np.argsort(fits)[::-1]
    alpha = deepcopy(pop[sorted_idx[0]])
    alpha_fit = fits[sorted_idx[0]]
    beta = deepcopy(pop[sorted_idx[1]]) if pop_size > 1 else deepcopy(alpha)
    delta = deepcopy(pop[sorted_idx[2]]) if pop_size > 2 else deepcopy(alpha)
    history = [alpha_fit]

    t0 = time.time()
    for it in range(max_iter):
        a = 2 - 2 * (it / max_iter)  # azalan keşif katsayısı

        for i in range(pop_size):
            seq, rots = pop[i]

            # Alpha, beta, delta'ya doğru hareket
            leaders = [alpha, beta, delta]
            chosen = leaders[np.random.randint(3)]

            if np.random.random() < a / 2:
                # Keşif
                new_seq = mutate_sequence(seq, strength=max(1, int(a)))
                new_rots = mutate_rotation(rots, strength=max(1, int(a)))
            else:
                # Sömürü — lidere doğru
                new_seq = crossover_order(seq, chosen[0])
                new_rots = crossover_rotation(rots, chosen[1])
                if np.random.random() < 0.2:
                    new_seq = mutate_sequence(new_seq)

            new_fit = fitness_fn(new_seq, new_rots)
            if new_fit > fits[i]:
                pop[i] = (new_seq, new_rots)
                fits[i] = new_fit

        # Alfa, beta, delta güncelle
        sorted_idx = np.argsort(fits)[::-1]
        if fits[sorted_idx[0]] > alpha_fit:
            alpha = deepcopy(pop[sorted_idx[0]])
            alpha_fit = fits[sorted_idx[0]]
        beta = deepcopy(pop[sorted_idx[1]]) if pop_size > 1 else deepcopy(alpha)
        delta = deepcopy(pop[sorted_idx[2]]) if pop_size > 2 else deepcopy(alpha)

        history.append(alpha_fit)

        if verbose and it % 500 == 0:
            print(f"  GWO iter {it}: {alpha_fit:.2f}%")

    return {
        "name": "GWO (Grey Wolf)",
        "best_fitness": alpha_fit,
        "best_solution": alpha,
        "history": history,
        "time": time.time() - t0,
    }


# ============================================================
# 8. TABU SEARCH
# ============================================================

def tabu_search(fitness_fn, n_pieces: int, max_iter: int = 250000,
                tabu_size: int = 100, verbose: bool = False) -> dict:
    """Tabu Search — toplam eval sayısı diğerleriyle eşit."""
    seq, rots = random_solution(n_pieces)
    cur_fit = fitness_fn(seq, rots)
    best_sol = (seq.copy(), rots.copy())
    best_fit = cur_fit
    history = [best_fit]
    tabu_list = []

    t0 = time.time()
    for it in range(max_iter):
        # Komşuluk: 5 aday üret
        candidates = []
        for _ in range(5):
            ns = mutate_sequence(seq, strength=1)
            nr = mutate_rotation(rots, strength=1)
            key = (tuple(ns), tuple(nr))
            if key not in tabu_list:
                nf = fitness_fn(ns, nr)
                candidates.append((ns, nr, nf, key))

        if not candidates:
            # Tabu boşsa rastgele atla
            seq, rots = random_solution(n_pieces)
            cur_fit = fitness_fn(seq, rots)
            continue

        # En iyi adayı seç
        candidates.sort(key=lambda x: x[2], reverse=True)
        seq, rots, cur_fit, key = candidates[0]

        tabu_list.append(key)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if cur_fit > best_fit:
            best_fit = cur_fit
            best_sol = (seq.copy(), rots.copy())

        history.append(best_fit)

        if verbose and it % 25000 == 0:
            print(f"  Tabu iter {it}: {best_fit:.2f}%")

    return {
        "name": "Tabu Search",
        "best_fitness": best_fit,
        "best_solution": best_sol,
        "history": history,
        "time": time.time() - t0,
    }
