"""

This script contains main driver class for the shockwave modelling.

:copyright:
    Srijan B. Das (sbdas@princeton.edu)
    Lucas Sawade (lsawade@princeton.edu)

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""




import numpy as np
import matplotlib
import scipy.integrate as integrate
import recon
import sys
class Config:
    """Class that handles parametrization of the domain.
    Boundary conditions etc."""

    def __init__(self,
                 dim=1,
                 cells=200,
                 CFL=0.1,
                 dx=1.0,
                 sigma=0.05,
                 v=1.0,
                 alpha=4.0,
                 profile="gaussian"):
        """
        This method handles initialization of the problem
        :param dim: Dimensions
        :param cells: Number of Finite Volume cells
        :param CFL: Stability condition
        :param dx: spacing
        :param sigma: sigma of the gaussian profile
        :param v: advection velocity
        :param alpha:
        :param profile: Initial profile before Advection, must be "gaussian",
                        "gauss_tophat" or an array of the same size as
                        N_{Cells} + 1

        """

        self.dim = dim
        self.cells = cells
        self.N = self.cells+1
        self.dx = dx
        self.CFL = CFL
        self.sigma = sigma
        self.v = v
        self.alpha = alpha
        self.N_ghost = 3
        # Compute cell edges
        self.x = np.arange(0, 1 + dx, dx)

        # Compute dt
        self.dt = CFL*dx/np.abs(v)

        # Set initial profile
        if profile == "gaussian":
            self.init_profile = self.init_profile_gaussian(self.x)
            self.profile_choice = "gaussian"

        elif profile == "gauss_tophat":
            self.init_profile = self.init_profile_gauss_tophat(self.x)
            self.profile_choice = "gauss_tophat"
        else:
            self.init_profile = profile
            self.profile_choice = "random"



    def init_profile_gaussian(self, x):
        """Function to define the initial closed form profile (non-averaged)"""
        return np.exp((-(x - 0.3) ** 2) / (2 * (self.sigma ** 2)))

    def init_profile_gauss_tophat(self, x):
        """Function constructing the tophat profile."""

        # below 60% of the profile set top hat
        profile = np.where(x <= 0.6, self.init_profile_gaussian(
            x), 0)

        # between 70 and 90 %
        profile = np.where(np.logical_and(x >= 0.7, x <= 0.9),
                           1, profile)

        return profile

    # def init_profile_gaussian(self, x):
    #     """Function to define the initial closed form profile (non-averaged)"""
    #     return np.exp((-(x - 0.7) ** 2) / (2 * (self.sigma ** 2)))

    # def init_profile_gauss_tophat(self, x):
    #     """Function constructing the tophat profile."""
    #
    #     # below 60% of the profile set top hat
    #     profile = np.where(x >= 0.4, self.init_profile_gaussian(
    #         x), 0)
    #
    #     # between 70 and 90 %
    #     profile = np.where(np.logical_and(x >= 0.1, x <= 0.3),
    #                        1, profile)
    #
    #     return profile




class FLASHM:
    """Handles running the main program."""

    def __init__(self, config, bc="periodic", method="first_order_upwind",
                 time_ep_method="SSPRK3", T=1):
        """
        :param config:
        :param bc: type of boundary conditions
        :param method: Flux reconstruction method
        :param time_ep_method: Time extrapolation method.
        :param T: Simulation duration
        """
        self.config = config
        self.bc = bc
        self.method = getattr(recon, method)
        self.time_ep_method = time_ep_method
        self.T = T
        self.t_step = 0
        self.phi = self.init_avg()

        # # Shift mat
        # self.s = []
        # self.s.append(np.array([-3, -2, -1, 0, 1, 2]))
        # self.s.append(-1 * s[0] + 1)


    def init_avg(self):
        """This method computes the initial average of the cells."""
        # Get x
        x = self.config.x

        # Initialize cell averages
        f_avg = np.zeros(self.config.cells)
        # averaging the function between cell edges
        # gaussian profile
        if self.config.profile_choice == "gaussian":
            for i in range(self.config.N - 1):
                dx = x[i + 1] - x[i]
                f_avg[i] = integrate.quad(self.config.init_profile_gaussian,
                                          x[i], x[i + 1])[0] / dx

        # gaussian and top hat
        elif self.config.profile_choice == "gauss_tophat":
            for i in range(self.config.N - 1):
                dx = x[i + 1] - x[i]
                f_avg[i] = integrate.quad(self.config.init_profile_gauss_tophat,
                                          x[i], x[i + 1])[0] / dx

        # Cell averages simply equal to the value of the profile.
        # This needs polishing by maybe figuring out a data vs. domain
        # interpolation or decimation scheme. For now, this shall hold
        else:
            f_avg = self.config.init_profile

        return f_avg

    def pad(self, phi, N_ghost):
        """Pads phi with 3 zeros at beginning and end. To make application of
        boundary conditions easier.
        :param phi: potential profile
        :param : potential profile"""

        return np.pad(phi, (N_ghost, N_ghost), "constant", constant_values=(0, 0))

    def apply_bc(self, phi):
        """
        Applies boundary conditions to a phi
        """

        # Set number of ghost cells
        N_ghost = 3

        # Initialize phi with ghost cells
        ghost_phi = self.pad(phi, N_ghost)

        # Apply BC's
        if self.bc == "periodic":
            ghost_phi[-N_ghost:] = phi[:N_ghost]
            ghost_phi[:N_ghost] = phi[-N_ghost:]
        elif self.bc == "fixed":
            ghost_phi[-N_ghost:] = 0
            ghost_phi[:N_ghost] = 0
        elif self.bc == "outgoing":
            """ 
            Use simple Newton extrapolation from the boundary:
            f(x) = f1 + (f2-f1)/(x2-x1)(x-x1)
            """
            # performing a linear extrapolation at the boundaries
            ghost_phi[0:N_ghost] = phi[0] + (phi[1] - phi[0]) \
                                   * - np.flip(np.arange(N_ghost) + 1, 0) \


            ghost_phi[-N_ghost:] = phi[-1] + (phi[-1] - phi[-2]) \
                                   * (1 + np.arange(N_ghost))

        return ghost_phi

    def lf_flux(self, phi):
        """ Handles computation of the left and right flux, i.e., basic flux
        splitting.
        :return: flux
        """

        # Shift mat
        s = np.zeros([4, 6], dtype=int)
        s[0] = np.array([-3, -2, -1, 0, 1, 2])
        s[1] = np.array([-3, -2, -1, 0, 1, 2])
        s[2] = (1 - s[0])
        s[3] = (1 - s[0] - 2)

        # For constant velocity, the flux and the parameters vary by just a
        # constant velocity. So, computing flux_discrete just once. For a
        # generic velocity, we need to compute twice

        # contains u^L_jph and u^L_jmh
        recon_L = self.method(self.config.x, self.apply_bc(phi),
                              self.config.v, self.config.N_ghost, s[0:2],
                              self.config.alpha)

        # contains u^R_jph and u^R_jmh
        recon_R = self.method(self.config.x, self.apply_bc(phi),
                              self.config.v, self.config.N_ghost, s[2:],
                              self.config.alpha)

        # Putting stuff together
        phi_jph_final = 0.5 * (self.config.v * recon_L[1, :]
                               + self.config.v * recon_R[1, :]) \
                        - 0.5 * np.abs(self.config.v) \
                        * (recon_R[1, :] - recon_L[1, :])

        phi_jmh_final = 0.5 * (self.config.v * recon_L[0, :]
                               + self.config.v * recon_R[0, :]) \
                        - 0.5 * np.abs(self.config.v) \
                        * (recon_R[0, :] - recon_L[0, :])

        flux = -(phi_jph_final - phi_jmh_final) / np.diff(self.config.x)

        return flux



    def one_time_step(self):
        """Advances the simulation one timestep."""


        # Parameters
        N_cells = self.config.cells
        dt = self.config.dt

        # Setting new Phi
        if self.t_step == 0:
            self.phi_new = self.phi

        phi_np1 = np.zeros(N_cells)
        phi1 = np.zeros(N_cells)
        phi2 = np.zeros(N_cells)

        phi1 = self.phi_new + dt * self.lf_flux(self.phi_new)

        phi2 = 0.75 * self.phi_new + 0.25 * (phi1 + dt * self.lf_flux(phi1))

        phi_np1 = (1.0 / 3.0) * self.phi_new \
                  + (2.0 / 3.0) * (phi2 + dt * self.lf_flux(phi2))

        # reassigning phi with the phi at next timestep
        self.phi_new = phi_np1

        # updating dt
        self.t_step += dt

        return self.phi_new

    def run(self):
        """ Runs the program a set amount of time
        """

        # Parameters
        N_cells = self.config.cells
        dt = self.config.dt
        N_ghost = self.config.N_ghost
        # Setting new Phi
        if self.t_step == 0:
            self.phi_new = self.phi

        while self.t_step < self.T:
            # Using the one time step function to advance one step in time.
            self.phi_new = self.one_time_step()

            # advance time itself.
            self.t_step += dt

        return self.phi_new



