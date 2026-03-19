"""8 optimizasyon algoritması v2 — continuous rotation + adaptive sigma."""
import numpy as np
from copy import deepcopy
import time


# ============================================================
# ORTAK YARDIMCILAR
# ============================================================

def random_solution(n: int) -> tuple:
    """Rastgele sıra + sürekli rotasyon üret."""
    seq = np.random.permutation(n).tolist()
    rots = np.random.uniform(0, 360, size=n).tolist()
    return seq, rots


def mutate_sequence(seq: list, strength: int = 1) -> list:
    """Sırada swap mutasyonu."""
    s = seq.copy()
    for _ in range(strength):
        i, j = np.random.choice(len(s), 2, replace=False)
        s[i], s[j] = s[j], s[i]
    return s


def mutate_rotation(rots: list, strength: int = 1, sigma: float = 30.0) -> list:
    """Gaussian perturbation ile rotasyon mutasyonu.

    Args:
        sigma: standart sapma (derece). Büyük = keşif, küçük = sömürü.
    """
    r = rots.copy()
    for _ in range(strength):
        i = np.random.randint(len(r))
        r[i] = (r[i] + np.random.normal(0, sigma)) % 360
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
    """Circular interpolation crossover — açı wraparound'ı doğru hesaplar."""
    alpha = np.random.uniform(0.3, 0.7)
    result = []
    for a, b in zip(r1, r2):
        diff = ((b - a + 180) % 360) - 180  # [-180, 180] aralığında fark
        result.append((a + alpha * diff) % 360)
    return result


# ============================================================
# 1. SPARROW SEARCH ALGORITHM (SSA)
# ============================================================

def sparrow_search(fitness_fn, n_pieces: int, pop_size: int = 50,
                   max_iter: int = 5000, verbose: bool = False) -> dict:
    PD = 0.2
    SD = 0.1
    n_disc = max(1, int(pop_size * PD))

    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    best_idx = np.argmax(fits)
    best_sol = deepcopy(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit]

    t0 = time.time()
    for it in range(max_iter):
        progress = it / max_iter
        R2 = np.random.random()

        # Adaptive sigma
        disc_sigma = 45.0 * (1 - progress)  # keşif: 45° → 0°
        expl_sigma = 5.0                     # sömürü: sabit 5°

        sorted_idx = np.argsort(fits)[::-1]

        # Discoverer'lar
        for i in range(n_disc):
            idx = sorted_idx[i]
            seq, rots = pop[idx]
            if R2 < 0.8:
                new_seq = mutate_sequence(seq, strength=max(1, int(3 * (1 - progress))))
                new_rots = mutate_rotation(rots, strength=max(1, int(2 * (1 - progress))), sigma=disc_sigma)
            else:
                new_seq = mutate_sequence(seq, strength=1)
                new_rots = mutate_rotation(rots, strength=1, sigma=expl_sigma)
            new_fit = fitness_fn(new_seq, new_rots)
            if new_fit > fits[idx]:
                pop[idx] = (new_seq, new_rots)
                fits[idx] = new_fit

        # Scrounger'lar
        worst_idx = sorted_idx[-1]
        for i in range(n_disc, pop_size):
            idx = sorted_idx[i]
            seq, rots = pop[idx]
            if fits[idx] > fits[worst_idx]:
                new_seq = crossover_order(seq, best_sol[0])
                new_rots = crossover_rotation(rots, best_sol[1])
            else:
                new_seq = mutate_sequence(seq, strength=3)
                new_rots = mutate_rotation(rots, strength=3, sigma=disc_sigma)
            new_fit = fitness_fn(new_seq, new_rots)
            if new_fit > fits[idx]:
                pop[idx] = (new_seq, new_rots)
                fits[idx] = new_fit

        # Scout
        n_scout = max(1, int(pop_size * SD))
        worst_indices = sorted_idx[-n_scout:]
        for idx in worst_indices:
            if np.random.random() < 0.5:
                pop[idx] = random_solution(n_pieces)
                fits[idx] = fitness_fn(*pop[idx])

        cur_best = np.argmax(fits)
        if fits[cur_best] > best_fit:
            best_fit = fits[cur_best]
            best_sol = deepcopy(pop[cur_best])

        history.append(best_fit)
        if verbose and it % 500 == 0:
            print(f"  SSA iter {it}: {best_fit:.2f}%")

    return {"name": "SSA (Sparrow Search)", "best_fitness": best_fit,
            "best_solution": best_sol, "history": history, "time": time.time() - t0}


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
        progress = it / max_iter
        sigma = 30.0 * (1 - progress) + 2.0  # 32° → 2°

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
            if np.random.random() < 0.3:
                child_rots = mutate_rotation(child_rots, sigma=sigma)

            new_pop.append((child_seq, child_rots))
            new_fits.append(fitness_fn(child_seq, child_rots))

        pop = new_pop[:pop_size]
        fits = new_fits[:pop_size]

        cur_best = np.argmax(fits)
        if fits[cur_best] > best_fit:
            best_fit = fits[cur_best]
            best_sol = deepcopy(pop[cur_best])

        history.append(best_fit)
        if verbose and it % 500 == 0:
            print(f"  GA iter {it}: {best_fit:.2f}%")

    return {"name": "GA (Genetic Algorithm)", "best_fitness": best_fit,
            "best_solution": best_sol, "history": history, "time": time.time() - t0}


# ============================================================
# 3. GA + SA (Hibrit Memetic)
# ============================================================

def ga_sa_hybrid(fitness_fn, n_pieces: int, pop_size: int = 50,
                 max_iter: int = 5000, sa_iters: int = 50, verbose: bool = False) -> dict:
    pop = [random_solution(n_pieces) for _ in range(pop_size)]
    fits = [fitness_fn(*ind) for ind in pop]

    best_idx = np.argmax(fits)
    best_sol = deepcopy(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit]

    t0 = time.time()
    for it in range(max_iter):
        progress = it / max_iter
        ga_sigma = 30.0 * (1 - progress) + 2.0

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
            if np.random.random() < 0.3:
                child_rots = mutate_rotation(child_rots, sigma=ga_sigma)

            new_pop.append((child_seq, child_rots))
            new_fits.append(fitness_fn(child_seq, child_rots))

        pop = new_pop[:pop_size]
        fits = new_fits[:pop_size]

        # --- SA adımı (her 50 iterasyonda en iyiye lokal arama) ---
        if it % 50 == 0 and it > 0:
            cur_seq, cur_rots = deepcopy(best_sol)
            cur_fit = best_fit
            temp = 10.0

            for sa_it in range(sa_iters):
                sa_sigma = max(1.0, 10.0 * temp / 10.0)  # 10° → 1°
                new_seq = mutate_sequence(cur_seq, strength=1)
                new_rots = mutate_rotation(cur_rots, strength=1, sigma=sa_sigma)
                new_fit = fitness_fn(new_seq, new_rots)

                delta = new_fit - cur_fit
                if delta > 0 or np.random.random() < np.exp(delta / max(temp, 0.01)):
                    cur_seq, cur_rots = new_seq, new_rots
                    cur_fit = new_fit

                temp *= 0.92

            if cur_fit > best_fit:
                best_fit = cur_fit
                best_sol = (cur_seq, cur_rots)
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

    return {"name": "GA+SA (Hibrit)", "best_fitness": best_fit,
            "best_solution": best_sol, "history": history, "time": time.time() - t0}


# ============================================================
# 4. SIMULATED ANNEALING (SA)
# ============================================================

def simulated_annealing(fitness_fn, n_pieces: int, max_iter: int = 250000,
                        verbose: bool = False) -> dict:
    seq, rots = random_solution(n_pieces)
    cur_fit = fitness_fn(seq, rots)
    best_sol = (seq.copy(), rots.copy())
    best_fit = cur_fit
    history = [best_fit]

    T0 = 50.0
    temp = T0
    cooling = 0.99995

    t0 = time.time()
    for it in range(max_iter):
        sigma = max(2.0, 45.0 * temp / T0)  # 45° → 2°
        new_seq = mutate_sequence(seq, strength=1 + int(temp > 10))
        new_rots = mutate_rotation(rots, strength=1, sigma=sigma)
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

    return {"name": "SA (Simulated Annealing)", "best_fitness": best_fit,
            "best_solution": best_sol, "history": history, "time": time.time() - t0}


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
    F = 0.8
    CR = 0.7
    sigma = 20.0  # DE: sabit sigma

    for it in range(max_iter):
        for i in range(pop_size):
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            seq_i, rots_i = pop[i]
            seq_a, rots_a = pop[a]
            seq_b, rots_b = pop[b]
            seq_c, rots_c = pop[c]

            trial_seq = seq_i.copy()
            trial_rots = rots_i.copy()

            for j in range(n_pieces):
                if np.random.random() < CR:
                    trial_seq = crossover_order(seq_a, seq_b) if np.random.random() < F else trial_seq
                    # DE differential: a + F*(b-c) for continuous angles
                    diff = ((rots_b[j] - rots_c[j] + 180) % 360) - 180
                    trial_rots[j] = (rots_a[j] + F * diff) % 360

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

    return {"name": "DE (Differential Evolution)", "best_fitness": best_fit,
            "best_solution": best_sol, "history": history, "time": time.time() - t0}


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
        w = 0.9 - 0.5 * (it / max_iter)
        sigma = 30.0 * w  # 27° → 12°

        for i in range(pop_size):
            seq, rots = pop[i]

            if np.random.random() < 0.5:
                seq = crossover_order(seq, pbest[i][0])
                rots = crossover_rotation(rots, pbest[i][1])

            if np.random.random() < 0.5:
                seq = crossover_order(seq, gbest[0])
                rots = crossover_rotation(rots, gbest[1])

            if np.random.random() < w:
                seq = mutate_sequence(seq)
                rots = mutate_rotation(rots, sigma=sigma)

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

    return {"name": "PSO (Particle Swarm)", "best_fitness": gbest_fit,
            "best_solution": gbest, "history": history, "time": time.time() - t0}


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
        a = 2 - 2 * (it / max_iter)
        sigma = 30.0 * a / 2  # 30° → 0.1°

        for i in range(pop_size):
            seq, rots = pop[i]
            leaders = [alpha, beta, delta]
            chosen = leaders[np.random.randint(3)]

            if np.random.random() < a / 2:
                new_seq = mutate_sequence(seq, strength=max(1, int(a)))
                new_rots = mutate_rotation(rots, strength=max(1, int(a)), sigma=sigma)
            else:
                new_seq = crossover_order(seq, chosen[0])
                new_rots = crossover_rotation(rots, chosen[1])
                if np.random.random() < 0.2:
                    new_seq = mutate_sequence(new_seq)
                    new_rots = mutate_rotation(new_rots, sigma=max(2.0, sigma))

            new_fit = fitness_fn(new_seq, new_rots)
            if new_fit > fits[i]:
                pop[i] = (new_seq, new_rots)
                fits[i] = new_fit

        sorted_idx = np.argsort(fits)[::-1]
        if fits[sorted_idx[0]] > alpha_fit:
            alpha = deepcopy(pop[sorted_idx[0]])
            alpha_fit = fits[sorted_idx[0]]
        beta = deepcopy(pop[sorted_idx[1]]) if pop_size > 1 else deepcopy(alpha)
        delta = deepcopy(pop[sorted_idx[2]]) if pop_size > 2 else deepcopy(alpha)

        history.append(alpha_fit)
        if verbose and it % 500 == 0:
            print(f"  GWO iter {it}: {alpha_fit:.2f}%")

    return {"name": "GWO (Grey Wolf)", "best_fitness": alpha_fit,
            "best_solution": alpha, "history": history, "time": time.time() - t0}


# ============================================================
# 8. TABU SEARCH
# ============================================================

def tabu_search(fitness_fn, n_pieces: int, max_iter: int = 250000,
                tabu_size: int = 200, verbose: bool = False) -> dict:
    seq, rots = random_solution(n_pieces)
    cur_fit = fitness_fn(seq, rots)
    best_sol = (seq.copy(), rots.copy())
    best_fit = cur_fit
    history = [best_fit]
    tabu_list = []

    sigma = 15.0  # Tabu: sabit orta sigma

    t0 = time.time()
    for it in range(max_iter):
        candidates = []
        for _ in range(5):
            ns = mutate_sequence(seq, strength=1)
            nr = mutate_rotation(rots, strength=1, sigma=sigma)
            # Tabu key: rounded angles for practical uniqueness
            key = (tuple(ns), tuple(round(r, 0) for r in nr))
            if key not in tabu_list:
                nf = fitness_fn(ns, nr)
                candidates.append((ns, nr, nf, key))

        if not candidates:
            seq, rots = random_solution(n_pieces)
            cur_fit = fitness_fn(seq, rots)
            continue

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

    return {"name": "Tabu Search", "best_fitness": best_fit,
            "best_solution": best_sol, "history": history, "time": time.time() - t0}
