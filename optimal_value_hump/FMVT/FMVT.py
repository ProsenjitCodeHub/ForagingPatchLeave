
import numpy as np
from numba import njit, prange

# ===============
# PARAMETERS
# ===============

N1 = 100
N = 15
tau = 100
foragers_per_patch = 3

regrowth_time = 500
steps = 100000
p = 0.01



Ef = 100
Es = 0.1

Et = 0.01

n_realizations = 10

L = N1 * N + (N1 - 1) * tau


all_mean_rates = []


# ===================
# PATCH POSITIONS
# ===================

patch_positions = []
for i in range(N1):
    for j in range(N1):
        x = i * (N + tau)
        y = j * (N + tau)
        patch_positions.append((x, y))


# ===========================
# Neighbours within a patch
# ===========================

@njit
def patch_neighbours(x, y, x0, y0, N):

    neigh = np.zeros((4, 2), dtype=np.int32)

    if not (x0 <= x < x0 + N and y0 <= y < y0 + N):
        return neigh, 0

    lx = x - x0
    ly = y - y0

    neigh[0, 0] = x0 + ((lx + 1) % N)
    neigh[0, 1] = y

    neigh[1, 0] = x0 + ((lx - 1) % N)
    neigh[1, 1] = y

    neigh[2, 0] = x
    neigh[2, 1] = y0 + ((ly + 1) % N)

    neigh[3, 0] = x
    neigh[3, 1] = y0 + ((ly - 1) % N)

    return neigh, 4

# ===========================
# NEIGHBOURS PATCHES
# ===========================

@njit
def neighbour_patches(i, j, N1, N, tau):

    neigh = np.zeros((4, 2), dtype=np.int32)

    patch_size = N + tau

    neigh[0, 0] = ((i + 1) % N1) * patch_size
    neigh[0, 1] = j * patch_size

    neigh[1, 0] = ((i - 1) % N1) * patch_size
    neigh[1, 1] = j * patch_size

    neigh[2, 0] = i * patch_size
    neigh[2, 1] = ((j + 1) % N1) * patch_size

    neigh[3, 0] = i * patch_size
    neigh[3, 1] = ((j - 1) % N1) * patch_size

    return neigh


# ====================
# UPDATE
# ====================

@njit
def one_step_update(food_sites, food_present, forager,
                    food_timer, time_spent, time_since_last_capture,
                    regrowth_time,
                    N, N1, tau,
                    food_eaten_agent,
                    patch_visits_agent,
                    energy_agent,food_in_patch_agent, new_forager, new_time):

    L = forager.shape[0]
    patch_size = N + tau

    #new_forager = np.zeros_like(forager, dtype=np.int32)
    #new_time = np.zeros_like(time_spent, dtype=np.int32)

    # count agents
    total_agents = 0
    for x in range(L):
        for y in range(L):
            if forager[x, y] > 0:
                total_agents += 1

    positions = np.empty((total_agents, 2), dtype=np.int32)

    idx = 0
    for x in range(L):
        for y in range(L):
            if forager[x, y] > 0:
                positions[idx, 0] = x
                positions[idx, 1] = y
                idx += 1

    np.random.shuffle(positions)

    for (x, y) in positions:

        agent = forager[x, y]
        forager[x, y] = -1

        time_since_last_capture[agent] += 1

        # --- identify patch ---
        x0 = (x // patch_size) * patch_size
        y0 = (y // patch_size) * patch_size

        i = x0 // patch_size
        j = y0 // patch_size

        nx, ny = x, y

        # ================= MOVE =================
        neigh, count = patch_neighbours(x, y, x0, y0, N)

        if count > 0:
            moved = False
            for _ in range(count):
                r = np.random.randint(count)
                tx, ty = neigh[r]

                if new_forager[tx, ty] == 0:
                    nx, ny = tx, ty
                    moved = True
                    break

            if not moved:
                nx, ny = x, y

        new_time_val = time_spent[x, y] + 1
        energy_agent[agent] -= Es

        # ================= move =================
        if new_forager[nx, ny] == 0:

            new_forager[nx, ny] = agent
            new_time[nx, ny] = new_time_val

            # ===== IMMEDIATE CONSUMPTION =====
            if food_sites[nx, ny] == 1 and food_present[nx, ny] == 1:

                food_present[nx, ny] = 0
                food_timer[nx, ny] = regrowth_time

                food_eaten_agent[agent] += 1
                energy_agent[agent] += Ef
                food_in_patch_agent[agent] += 1

                dt = time_since_last_capture[agent]
                marginal_rate = 1.0 / (dt + 1e-8)

                # ===== compute average rate =====
                avg_rate = (food_in_patch_agent[agent]) / (new_time_val + tau + 1e-8)

                # ===== reset timer =====
                time_since_last_capture[agent] = 0

                # ===== LEAVING DECISION =====
                if marginal_rate <= avg_rate:

                    food_in_patch_agent[agent] = 0

                    patch_visits_agent[agent] += 1
                    energy_agent[agent] -= Et * tau

                    patch_neigh = neighbour_patches(i, j, N1, N, tau)

                    r = np.random.randint(4)
                    px = patch_neigh[r, 0]
                    py = patch_neigh[r, 1]

                    nx = np.random.randint(px, px + N)
                    ny = np.random.randint(py, py + N)

                    new_forager[nx, ny] = agent
                    new_time[nx, ny] = 0

                    continue

        else:
            # stay if blocked
            new_forager[x, y] = agent
            new_time[x, y] = time_spent[x, y] + 1

            if food_sites[x, y] == 1 and food_present[x, y] == 1:

                food_present[x, y] = 0
                food_timer[x, y] = regrowth_time

                food_eaten_agent[agent] += 1
                energy_agent[agent] += Ef
                food_in_patch_agent[agent] += 1


                dt = time_since_last_capture[agent]
                marginal_rate = 1.0 / (dt + 1e-8)

                avg_rate = food_in_patch_agent[agent]/(new_time[x,y]+tau+1e-8)

                time_since_last_capture[agent] = 0

                if marginal_rate <= avg_rate:

                    food_in_patch_agent[agent] = 0

                    patch_visits_agent[agent] += 1
                    energy_agent[agent] -= Et * tau

                    patch_neigh = neighbour_patches(i, j, N1, N, tau)

                    r = np.random.randint(4)
                    px = patch_neigh[r, 0]
                    py = patch_neigh[r, 1]

                    nx = np.random.randint(px, px + N)
                    ny = np.random.randint(py, py + N)

                    new_forager[nx, ny] = agent
                    new_time[nx, ny] = 0

                    continue

    # ================= UPDATE STATE =================
    forager[:, :] = new_forager
    time_spent[:, :] = new_time

    # ================= REGROWTH =================
    for x in range(L):
        for y in range(L):

            if food_timer[x, y] > 0:
                food_timer[x, y] -= 1

                if food_timer[x, y] == 0 and food_sites[x, y] == 1:
                    food_present[x, y] = 1


# ===========================
# PARALLEL REALIZATIONS
# ===========================

@njit(parallel=True)
def run_parallel():

    results = np.zeros(n_realizations)

    for run in prange(n_realizations):

        np.random.seed(run + 1234)

        food_sites = np.zeros((L, L), dtype=np.int32)
        food_present = np.zeros((L, L), dtype=np.int32)
        forager = np.zeros((L, L), dtype=np.int32)
        food_timer = np.zeros((L, L), dtype=np.int32)
        time_spent = np.zeros((L, L), dtype=np.int32)
        new_forager = np.zeros((L, L), dtype=np.int32)
        new_time = np.zeros((L, L), dtype=np.int32)

        # initialize food
        for i in range(N1):
            for j in range(N1):

                x0 = i * (N + tau)
                y0 = j * (N + tau)

                F = np.random.binomial(N*N,p) # Binomial distribution with mean= N^2p

                # lam = N * N * p   # same mean  
                # F = np.random.poisson(lam)  # Poisson distribution

                # mu = N * N * p
                # disp=1
                # p_nb = disp / (disp + mu)
                # F = np.random.negative_binomial(disp, p_nb)  # Negative Binomial distribution

                F = min(F, N*N)

                for _ in range(F):
                    x = x0 + np.random.randint(N)
                    y = y0 + np.random.randint(N)
                    food_sites[x, y] = 1
                    food_present[x, y] = 1

        # initialize foragers
        agent_id = 1
        for i in range(N1):
            for j in range(N1):

                x0 = i * (N + tau)
                y0 = j * (N + tau)

                for _ in range(foragers_per_patch):
                    x = x0 + np.random.randint(N)
                    y = y0 + np.random.randint(N)
                    forager[x, y] = agent_id
                    agent_id += 1

        num_agents = agent_id

        food_eaten_agent = np.zeros(num_agents)
        patch_visits_agent = np.zeros(num_agents)
        energy_agent = np.zeros(num_agents)
        food_in_patch_agent = np.zeros(num_agents)
        time_since_last_capture = np.zeros(num_agents)

        # simulate
        for t in range(steps):
            one_step_update(food_sites, food_present, forager,
                            food_timer, time_spent, time_since_last_capture,
                            regrowth_time,
                            N, N1, tau,
                            food_eaten_agent,
                            patch_visits_agent,
                            energy_agent,
                            food_in_patch_agent, new_forager, new_time)

        rates = (food_eaten_agent * Ef - Es * steps - Et*tau * patch_visits_agent) / (steps + tau * patch_visits_agent)

        results[run] = np.mean(rates[1:])

    return results

# ===========================
# MAIN
# ===========================

results = run_parallel()

R_star = np.mean(results)
std_R = np.std(results)

np.savetxt("FMVT_summary.txt", np.array([R_star, std_R]))

