from flashm import Config, FLASHM

import numpy as np
import matplotlib.pyplot as plt

def main():
    N_cells = 200       # number of cells
    N = N_cells + 1     # N+1 is the number of cell edges.
    CFL = 0.5           # CFL number
    dx = 1.0 / N_cells  # grid spacing
    sig = 0.05          # sigma for the gaussian in the initial function
    v = 1.0             # advection velocity
    T = 1/v           # Length of domain in code units is 1.0
    alpha = 4.0         # parameter for defining the MC limited

    # Setting Configuration.
    config = Config(dim=1, cells=N_cells, CFL=CFL, dx=dx, sigma=sig, v=v,
                    alpha=alpha, profile="gauss_tophat")


    # Initalize the solver
    bc = "periodic"
    time_stepping = "SSPRK3"
    reconstruction_method = "first_order_upwind"
    reconstruction_method = "second_order_centered"

    flashm = FLASHM(config, bc=bc, method=reconstruction_method,
                 time_ep_method=time_stepping, T=T)

    # plot
    plt.ion()
    plt.figure()
    t = 0
    while t<T:
        # Run the solver
        phi_new = flashm.one_time_step()

        plt.title(str(t))
        plt.plot(config.x[1:], flashm.init_avg(), label="Initial profile")
        plt.plot(config.x[1:], phi_new, label="Profile after time T")
        # plt.plot(config.x[1:], phi_new-flashm.init_avg(), label="Profile after time T")
        plt.legend()
        plt.pause(0.001)
        plt.clf()

        t += config.dt

if __name__ == "__main__":
    main()
