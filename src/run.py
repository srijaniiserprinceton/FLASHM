from flashm import Config, FLASHM

import numpy as np
import matplotlib.pyplot as plt
import time


def velocity_profile(x):
    """creates velocity profile
    """

    # return np.ones(len(x))*-1
    # return  3 + np.cos(x*2*np.pi) + np.sin(x*10*np.pi)
    # return np.sqrt(x) + np.sqrt(1 - x)
    # return -np.sqrt(x) + np.sqrt(1 - x)
    # return  3*np.cos(x*4*np.pi) + np.sin(x*10*np.pi)
    return np.sin(x*2*np.pi)

def main():
    N_cells = 200       # number of cells
    N = N_cells + 1     # N+1 is the number of cell edges.
    CFL = 0.5           # CFL number
    dx = 1.0 / N_cells  # grid spacing
    sig = 0.05          # sigma for the gaussian in the initial function
    alpha = 4.0         # parameter for defining the MC limited

    x = np.arange(0, 1+dx, dx)
    v = velocity_profile(x) # Advection profile
    T = 1  # Length of time of the simulation in code units is 1.0

    # Setting Configuration.
    config = Config(dim=1, cells=N_cells, CFL=CFL, dx=dx, sigma=sig, v=v,
                    alpha=alpha, profile="gauss_tophat")



    # Initalize the solver
    bc = "outgoing"

    time_stepping = "SSPRK3"

    # reconstruction_method = "first_order_upwind"
    # reconstruction_method = "second_order_centered"
    # reconstruction_method = "third_order_upwind"
    # reconstruction_method = "MC"
    reconstruction_method = "MP5"

    flashm = FLASHM(config, bc=bc, method=reconstruction_method,
                 time_ep_method=time_stepping, T=T, evolve_v=fancy)

    # plot
    plt.ion()
    plt.figure()
    # plt.ylim([-1.5, 2.5])
    plt.xlim([0, 1])
    t = 0

    counter = 0
    t1 = np.arange(0, T+config.dt, config.dt)

    n_plots = np.round(len(t1)/100)

    energy = np.zeros(len(t1))
    energy[0] = np.sum(flashm.phi * np.diff(config.x))
    rel_energy = energy[0]
    energy[0] = 1

    start = time.time()

    while t<T-config.dt:
        # Run the solver
        phi_new = flashm.one_time_step()

        counter += 1
        energy[counter] = np.sum(phi_new * np.diff(config.x)) / rel_energy

        if counter%n_plots:
            # plt.ylim([-.5, 1.5])
            # plt.xlim([0, 1])
            plt.title("%2.3f s" % t)
            plt.plot(config.x[1:], flashm.init_avg(), label="Initial profile")
            plt.plot(config.x[1:], phi_new, label="Profile after time T")
            plt.plot(t1[:counter+1], energy[:counter+1], label="Energy")
            plt.plot(config.x, config.v, label="Velocity", zorder=2)
            plt.legend(loc=2)
            plt.pause(0.001)
            plt.clf()

        t += config.dt

    end = time.time()
    print("Time elapsed: %3.1f" %(end-start))

    # Run the solver
    plt.ylim([-1.5, 5])
    plt.xlim([0, 1])


    plt.title("%2.3f s" % t)
    plt.plot(config.x[1:], flashm.init_avg(), label="Initial profile", zorder=4)
    plt.plot(config.x[1:], phi_new, label="Profile after time T=%1.1fs" % t,
             linewidth=3, zorder=5)
    plt.plot(t1[:counter + 1], energy[:counter + 1], label="$Rel. E_{tot}$ ",
             zorder=3)
    plt.plot(config.x, flashm.config.v, label="Velocity", zorder=2)

    plt.legend(loc=0)
    plt.show(block=True)


if __name__ == "__main__":
    main()

