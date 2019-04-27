"""

This script contains the reconstruction algorithms.

:copyright:
    Srijan B. Das (sbdas@princeton.edu)
    Lucas Sawade (lsawade@princeton.edu)

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""

import numpy as np

def minmod(x,y):
    return 0.5*(np.sign(x) + np.sign(y))*np.minimum(np.abs(x), np.abs(y))


def second_order_centered(x, phi, v, N_ghost):
    """ Computes the flux using the 2nd order centered scheme.
    The phi that is taken in is padded. The padding depends on the
    boundary conditions."""

    flux = -v * (phi[N_ghost+1:-N_ghost+1] - phi[N_ghost-1:-N_ghost-1]) / (2 * np.diff(x))
    return flux


def first_order_upwind(x, phi, v, N_ghost):
    flux = -v * (phi[N_ghost:-N_ghost] - phi[N_ghost-1:-N_ghost-1]) / np.diff(x)
    return flux


def third_order_upwind(x, phi, v, N_ghost):
    flux = -v * (
            phi[N_ghost:-N_ghost] / 2.0 + phi[N_ghost+1:-N_ghost+1] / 3.0
            - phi[N_ghost-1:-N_ghost-1]
            + phi[N_ghost-2:-N_ghost-2] / 6.0) / np.diff(x)
    return flux


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