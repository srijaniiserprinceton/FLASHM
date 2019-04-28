from flashm import Config, FLASHM

import numpy as np
import matplotlib.pyplot as plt

def main():
    N_cells = 200       # number of cells
    N = N_cells + 1     # N+1 is the number of cell edges.
    CFL = 0.25           # CFL number
    dx = 1.0 / N_cells  # grid spacing
    sig = 0.05          # sigma for the gaussian in the initial function
    v = -1.0             # advection velocity
    T = 1/np.abs(v)           # Length of domain in code units is 1.0
    alpha = 4.0         # parameter for defining the MC limited

    # Setting Configuration.
    config = Config(dim=1, cells=N_cells, CFL=CFL, dx=dx, sigma=sig, v=v,
                    alpha=alpha, profile="gauss_tophat")


    # Initalize the solver
    bc = "outgoing"

    time_stepping = "SSPRK3"

    # reconstruction_method = "first_order_upwind"
    # reconstruction_method = "second_order_centered"
    # reconstruction_method = "third_order_upwind"
    reconstruction_method = "MC"
    # reconstruction_method = "MP5"

    flashm = FLASHM(config, bc=bc, method=reconstruction_method,
                 time_ep_method=time_stepping, T=T)

    # plot
    plt.ion()
    plt.figure()
    plt.ylim([-.5, 1.5])
    plt.xlim([0, 1])
    t = 0


    # Start countin'
    start = time.time()

    t1 = np.arange(0, 1+config.dt, config.dt)
    energy = np.zeros(len(t1))
    energy[0] = np.linalg.norm(flashm.phi)
    rel_energy = energy[0]
    energy[0] = energy[0]/energy[0]
    counter = 0

    while t<T-config.dt:
        # Run the solver
        phi_new = flashm.one_time_step()

        counter += 1
        energy[counter] = np.linalg.norm(phi_new)/rel_energy
        if counter%10:
            plt.ylim([-.5, 1.5])
            plt.xlim([0, 1])
            plt.title("%2.3f s" % t)
            plt.plot(config.x[1:], flashm.init_avg(), label="Initial profile")
            plt.plot(config.x[1:], phi_new, label="Profile after time T")
            plt.plot(t1[:counter+1], energy[:counter+1], label="Energy")
            # plt.plot(config.x[1:], phi_new-flashm.init_avg(), label="Profile after time T")
            plt.legend(loc=2)
            plt.pause(0.001)
            plt.clf()

        t += config.dt


    # Run the solver
    plt.ylim([-.5, 1.5])
    plt.xlim([0, 1])


    plt.title("%2.3f s" % t)
    plt.plot(config.x[1:], flashm.init_avg(), label="Initial profile")
    plt.plot(config.x[1:], phi_new, label="Profile after time T")
    # plt.plot(config.x[1:], phi_new-flashm.init_avg(), label="Profile after time T")
    plt.legend(loc=2)
    plt.show(block=True)


if __name__ == "__main__":
    main()

