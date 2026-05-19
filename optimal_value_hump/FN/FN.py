
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

L = N1 * N + (N1 - 1) * tau

Ef = 10
Es = 1

Et = 0.01

n_realizations = 10

# ===================
# PATCH POSITIONS
# ===================

patch_positions = np.zeros((N1*N1, 2), dtype=np.int32)
idx = 0
for i in range(N1):
    for j in range(N1):
        patch_positions[idx,0] = i * (N + tau)
        patch_positions[idx,1] = j * (N + tau)
        idx += 1

# ===========================
# NEIGHBOURS WITHIN PATCH
# ===========================

@njit
def patch_neighbours(x, y, x0, y0, N):

    neigh = np.zeros((4, 2), dtype=np.int32)

    if not (x0 <= x < x0 + N and y0 <= y < y0 + N):
        return neigh, 0

    lx = x - x0
    ly = y - y0

    neigh[0,0] = x0 + ((lx + 1) % N)
    neigh[0,1] = y

    neigh[1,0] = x0 + ((lx - 1) % N)
    neigh[1,1] = y

    neigh[2,0] = x
    neigh[2,1] = y0 + ((ly + 1) % N)

    neigh[3,0] = x
    neigh[3,1] = y0 + ((ly - 1) % N)

    return neigh, 4


# ===========================
# NEIGHBOURS PATCHES
# ===========================

@njit
def neighbour_patches(i, j, N1, N, tau):

    neigh = np.zeros((4, 2), dtype=np.int32)
    ps = N + tau

    neigh[0,0] = ((i + 1) % N1) * ps
    neigh[0,1] = j * ps

    neigh[1,0] = ((i - 1) % N1) * ps
    neigh[1,1] = j * ps

    neigh[2,0] = i * ps
    neigh[2,1] = ((j + 1) % N1) * ps

    neigh[3,0] = i * ps
    neigh[3,1] = ((j - 1) % N1) * ps

    return neigh


# =========================================================
# ONE STEP UPDATE
# =========================================================

@njit
def one_step_update(food_sites, food_present, forager,
                    food_timer, food_in_patch,
                    regrowth_time, F_N,
                    N, N1, tau,
                    food_eaten_agent,
                    patch_visits_agent,
                    energy_agent):

    L = forager.shape[0]
    patch_size = N + tau

    new_forager = np.zeros_like(forager)
    new_food = np.zeros_like(food_in_patch)

    # count agents
    total_agents = 0
    for x in range(L):
        for y in range(L):
            if forager[x, y] > 0:
                total_agents += 1

    # ================= RANDOM SEQUENTIAL UPDATE =================

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

        # --- identify patch ---
        x0 = (x // patch_size) * patch_size
        y0 = (y // patch_size) * patch_size

        i = x0 // patch_size
        j = y0 // patch_size

        nx, ny = x, y

        # ================= LEAVE =================
        leave_flag = False

        if food_in_patch[x, y] >= F_N:

            leave_flag = True


            patch_visits_agent[agent] += 1
            energy_agent[agent] -= Et * tau 

            patch_neigh = neighbour_patches(i, j, N1, N, tau)

            r = np.random.randint(4)
            px = patch_neigh[r, 0]
            py = patch_neigh[r, 1]

            nx = np.random.randint(px, px + N)
            ny = np.random.randint(py, py + N)


        # ================= STAY =================
        else:

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


            energy_agent[agent] -= Es

        # ================= move =================
        if new_forager[nx, ny] == 0:

            new_forager[nx, ny] = agent
            if leave_flag:
                new_food[nx, ny] = 0
            else:
                new_food[nx, ny] = food_in_patch[x, y]    

            # ===== IMMEDIATE CONSUMPTION =====
            if food_sites[nx, ny] == 1 and food_present[nx, ny] == 1:
                food_present[nx, ny] = 0
                food_timer[nx, ny] = regrowth_time
                food_eaten_agent[agent] += 1
                energy_agent[agent] += Ef
                new_food[nx, ny] += 1

        else:
            # stay if blocked
            new_forager[x, y] = agent
            new_food[x, y] = food_in_patch[x, y]

            # consume if staying
            if food_sites[x, y] == 1 and food_present[x, y] == 1:
                food_present[x, y] = 0
                food_timer[x, y] = regrowth_time
                food_eaten_agent[agent] += 1
                energy_agent[agent] += Ef

                new_food[x, y] = food_in_patch[x, y] + 1


    # ================= UPDATE STATE =================
    forager[:, :] = new_forager
    food_in_patch[:, :] = new_food

    # ================= REGROWTH =================
    for x in range(L):
        for y in range(L):

            if food_timer[x, y] > 0:
                food_timer[x, y] -= 1

                if food_timer[x, y] == 0 and food_sites[x, y] == 1:
                    food_present[x, y] = 1


# =========================================================
# PARALLEL REALIZATIONS
# =========================================================

@njit(parallel=True)
def run_parallel(F_N):

    results = np.zeros(n_realizations)

    for run in prange(n_realizations):

        # independent RNG per thread
        seed = run + 1234
        np.random.seed(seed)

        food_sites = np.zeros((L, L), dtype=np.int32)
        food_present = np.zeros((L, L), dtype=np.int32)
        forager = np.zeros((L, L), dtype=np.int32)
        food_timer = np.zeros((L, L), dtype=np.int32)
        food_in_patch = np.zeros((L, L), dtype=np.int32)

        # initialize food
        for k in range(patch_positions.shape[0]):
            x0 = patch_positions[k,0]
            y0 = patch_positions[k,1]
            

            F = np.random.binomial(N*N,p) # Binomial distribution with mean= N^2p

            # lam = N * N * p # same mean  
            # F = np.random.poisson(lam)  # Poisson distribution

            # mu = N * N * p
            # disp=1
            # p_nb = disp / (disp + mu)
            # F = np.random.negative_binomial(disp, p_nb)  # Negative Binomial distribution

            F = min(F, N*N)

            coords = [(x, y) for x in range(x0, x0+N)
                             for y in range(y0, y0+N)]

            if F > 0:
                chosen_idx = np.random.choice(len(coords), F, replace=False)
                for idx in chosen_idx:
                    x, y = coords[idx]
                    food_sites[x,y] = 1
                    food_present[x,y] = 1                 

        # initialize agents
        agent_id = 1
        for k in range(patch_positions.shape[0]):
            x0 = patch_positions[k,0]
            y0 = patch_positions[k,1]

            for x in range(x0, x0+N):
                for y in range(y0, y0+N):
                    if agent_id <= N1*N1*foragers_per_patch:
                        forager[x,y] = agent_id
                        agent_id += 1

        num_agents = agent_id

        food_eaten_agent = np.zeros(num_agents, dtype=np.int32)
        patch_visits_agent = np.zeros(num_agents, dtype=np.int32)
        energy_agent = np.zeros(num_agents)

        # simulate
        for t in range(steps):
            one_step_update(food_sites, food_present, forager,
                            food_timer, food_in_patch,
                            regrowth_time, F_N,
                            N, N1, tau,
                            food_eaten_agent,
                            patch_visits_agent,
                            energy_agent)

        rates = (food_eaten_agent * Ef - Es * steps - Et*tau * patch_visits_agent) / (steps + tau * patch_visits_agent)

        results[run] = np.mean(rates[1:])

    return np.mean(results)


# =========================================================
# MAIN LOOP
# =========================================================

F_values = np.arange(1, 250, 2)
results = []

for F_N in F_values:

    print(f"\nRunning F_N = {F_N}")

    mean_rate = run_parallel(F_N)

    print(f"F_N = {F_N}, Mean rate = {mean_rate}")

    results.append([F_N, mean_rate])

results = np.array(results)

np.savetxt("F_N_vs_rate.txt", results,
           header="F_N  Mean_rate")

