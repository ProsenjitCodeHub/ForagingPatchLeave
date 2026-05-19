import numpy as np
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'notebook'])


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "axes.labelsize": 22,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

# =========================
# LOAD DATA
# =========================

data = np.loadtxt("Tleave_vs_rate.txt", skiprows=1)

data = data[::7]

T = data[:, 0]
R = data[:, 1]


idx = np.argsort(T)
T = T[idx]
R = R[idx]



COLOR_DATA = "black"
COLOR_OPT  = "#2a9d8f"

# =========================
# FIND MAX
# =========================
idx_max = np.argmax(R)
T_opt = T[idx_max]
R_opt = R[idx_max]

# =========================
# PLOT
# =========================

plt.figure(figsize=(6,4))

plt.plot(T, R, '*',
         color=COLOR_DATA,
         markersize=7,
         markeredgewidth=1)


# joining line
plt.plot(
    T,
    R,
    '-',
    color=COLOR_DATA,
    linewidth=2,
    alpha=0.8
)

plt.vlines(T_opt, 0, R_opt,
           linestyles=':',
           linewidth=2,
           color=COLOR_OPT)

# mark max point
plt.plot(T_opt, R_opt,
         '*',
         color=COLOR_OPT,
         markersize=12)

plt.xlabel('Time on patch')
#plt.ylabel('Energy gain rate')

# =========================
# AXIS ROUNDING
# =========================
xmax = np.ceil(T.max()/50)*50
ymax = np.ceil(R.max()/0.2)*0.2

plt.xlim(T.min(), xmax)
plt.ylim(0, ymax)

plt.yticks([0.0, 0.2, 0.4,0.6,0.8])
plt.xticks([0,100,200,300,400,500])

plt.tight_layout()
plt.grid(False)

plt.savefig("FT.png", dpi=300)
plt.show()
