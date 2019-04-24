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

        # Compute cell edges
        self.x = np.arange(0, 1 + dx, dx)

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


    def init_profile(self):
        """Computes the initial profile of none is given."""

    def init_profile_gaussian(self, x):
        """Function to define the initial closed form profile (non-averaged)"""
        return np.exp((-(x - 0.3) ** 2) / (2 * (self.sigma ** 2)))

    def init_profile_gauss_tophat(self, x):
        """Function constructing the tophat profile."""

        # below 60% of the profile set top hat
        profile = np.where(x <= np.max(x)*0.6, self.init_profile_gaussian(x), 0)

        # between 70 and 90 %
        profile = np.where(np.logical_and(np.where(x >= 0.7),
                                          np.where(x <= 0.9)),
                           1, profile)

        return profile



class FLASHM:
    """Handles running the main program."""

    def __init__(self, config, bc="periodic", method="first_order_upwind",
                 time_ep_method="SSPRK3", T=1,):
        """
        :param config:
        :param bc: type of boundary conditions
        :param method: Flux reconstruction method
        :param time_ep_method: Time extrapolation method.
        :param T: Simulation duration
        """
        self.config = config
        self.bc = "periodic"
        self.method = method
        self.time_ep_method = time_ep_method
        self.T = T
        self.phi = self.init_avg()

    def init_avg(self):
        """This method computes the initial average of the cells."""
        # Get x
        x = self.config.x

        # Initialize cell averages
        f_avg = np.zeros(self.config.cells)

        # averaging the function between cell edges
        # gaussian profile
        if self.config.profile_choice == "gaussian":
            for i in range(self.N - 1):
                dx = x[i + 1] - x[i]
                f_avg[i] = integrate.quad(self.config.init_profile_gaussian,
                                          self.x[i], x[i + 1]) / dx

        # gaussian and top hat
        elif self.config.profile_choice == "gauss_tophat":
            for i in range(self.N - 1):
                dx = x[i + 1] - x[i]
                f_avg[i] = integrate.quad(self.config.init_profile_gauss_tophat,
                                          x[i], x[i + 1]) / dx

        # Cell averages simply equal to the value of the profile.
        # This needs polishing by maybe figuring out a data vs. domain
        # interpolation or decimation scheme. For now, this shall hold
        else:
            f_avg = self.config.init_profile

        return f_avg

    def pad_phi(self):
        """Pads phi with 3 zeros at beginning and end. To make application of
        boundary conditions easier."""

        return np.pad(self.phi, (3, 3), "constant", constant_values=(0, 0))

    def apply_boundary_conditions(self):
        """
        Applies boundary conditions to phi
        """

        # Initialize phi with ghost cells
        self.ghost_phi = pad_phi()

        # Apply BC's
        if self.bc == "periodic":
            self.ghost_phi[-3:] = self.phi[-3]
            self.ghost_phi[:3] = self.phi[:3]


    def second_order_centered(self):
        """ Computes the flux using the 2nd order centered scheme.
        The phi that is taken in is padded. The padding depens on the
        boundary conditions."""
        flux = -self.config.v * (self.phi[4:] - phi) / (2 * dx)

        return flux

    def first_order_upwind(self):
        flux = -v * (phi - phi_jm1) / dx

    def third_order_upwind(self):
        flux = -v * (
                    phi / 2.0 + phi_jp1 / 3.0 - phi_jm1 + phi_jm2 / 6.0) / dx
    # MC limiter scheme
    def MC(self):
        phi_jp1_MP = phi + minmod_two(phi_jp1 - phi,
                                      alpha * (phi - phi_jm1))
        phi_jm1_MP = phi_jm1 + minmod_two(phi - phi_jm1,
                                          alpha * (phi_jm1 - phi_jm2))

        phi_jp1_3u = (5.0 / 6.0) * phi + (1.0 / 3.0) * phi_jp1 - (
                    1.0 / 6.0) * phi_jm1
        phi_jm1_3u = (5.0 / 6.0) * phi_jm1 + (1.0 / 3.0) * phi - (
                    1.0 / 6.0) * phi_jm2

        phi_jphalf_MC = phi + minmod_two(phi_jp1_3u - phi, phi_jp1_MP - phi)
        phi_jmhalf_MC = phi_jm1 + minmod_two(phi_jm1_3u - phi_jm1,
                                             phi_jm1_MP - phi_jm1)

        flux = -v * (phi_jphalf_MC - phi_jmhalf_MC) / dx

    # Suresh-Hyunh scheme
    def MP5(self):
        """
        This method computes the MP5
        :return: flux
        """
        phi_j_final = np.zeros([2, N_cells])
        # print(np.shape(phi_j_final))

        # j = 2 is the j-half case and j = 3 is the j+half case
        for j in [2, 3]:
            # This is the MP5 original reconstruction
            phi_jph_orig = (2.0 * phi_master[j - 2] - 13.0 * phi_master[
                j - 1] + 47.0 * phi_master[j] + 27 * phi_master[
                                j + 1] - 3.0 * phi_master[
                                j + 2]) / 60.0  # for right-going wind

            # phi_jph_orig = (7.0/12.0)*(phi_master[j] + phi_master[j+1]) - (1.0/12.0)*(phi_master[j-1] + phi_master[j+2]) #using Collela 4th ordered centered

            phi_MP = phi_master[j] + minmod(np.array(
                [phi_master[j + 1] - phi_master[j],
                 alpha * (phi_master[j] - phi_master[j - 1])]))

            # creating the array to be compared with epsilon at each interface
            cond_array = np.multiply(phi_jph_orig - phi_master[j],
                                     phi_jph_orig - phi_MP)

            # performing simple assignment if the condition is satisfied
            phi_j_final[j - 2, cond_array <= eps_SH] = phi_jph_orig[
                cond_array <= eps_SH]

            # further computation if the condition is not satisfied
            d = phi_master[j - 1] + phi_master[j + 1] - 2 * phi_master[j]
            dp1 = phi_master[j] + phi_master[j + 2] - 2 * phi_master[j + 1]
            dm1 = phi_master[j - 2] + phi_master[j] - 2 * phi_master[j - 1]

            d_M4_jph = minmod(
                np.array([4.0 * d - dp1, 4.0 * dp1 - d, d, dp1]))
            d_M4_jmh = minmod(
                np.array([4.0 * dm1 - d, 4.0 * d - dm1, dm1, d]))

            phi_UL = phi_master[j] + alpha * (
                        phi_master[j] - phi_master[j - 1])

            phi_AV = 0.5 * (phi_master[j] + phi_master[j + 1])

            phi_MD = phi_AV - 0.5 * d_M4_jph

            phi_LC = phi + 0.5 * (phi_master[j] - phi_master[j - 1]) + (
                        4.0 / 3.0) * d_M4_jmh

            phi_min = np.maximum(
                np.minimum(phi_master[j], phi_master[j + 1], phi_MD),
                np.minimum(phi_master[j], phi_UL, phi_LC))
            phi_max = np.minimum(
                np.maximum(phi_master[j], phi_master[j + 1], phi_MD),
                np.maximum(phi_master[j], phi_UL, phi_LC))

            # for the interfaces where the condition array did not satisfy the condition
            phi_j_final[j - 2, cond_array > eps_SH] = (
                        phi_jph_orig + minmod(np.array(
                    [phi_min - phi_jph_orig, phi_max - phi_jph_orig])))[
                cond_array > eps_SH]

        flux = -v * (phi_j_final[1, :] - phi_j_final[0,
                                         :]) / dx  # the flux without Reimann solver. Not considering waves in both direction


