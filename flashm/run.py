from .flashm import Config

import numpy as np
import matplotlib.pyplot as plt

N_cells = 200       # number of cells
N = N_cells + 1     # N+1 is the number of cell edges.
CFL = 0.1           # CFL number
dx = 1.0 / N_cells  # grid spacing
sig = 0.05          # sigma for the gaussian in the initial function
v = 1.0             # advection velocity
T = 1 / v           # Length of domain in code units is 1.0
alpha = 4.0         # parameter for defining the MC limited

# Setting Configuration.
config = Config(dim=1, cells=N_cells, CFL=0.1, dx=dx, sigma=sig, v=1.0,
                alpha=4.0, profile="gauss_tophat")


