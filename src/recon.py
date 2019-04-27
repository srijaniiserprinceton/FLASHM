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


def minmod(x, y):
    return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))


def second_order_centered(x, phi, v, N_ghost, alpha):
    """ Computes the flux using the 2nd order centered scheme.
    The phi that is taken in is padded. The padding depends on the
    boundary conditions."""

    flux = -v * (phi[N_ghost + 1:-N_ghost + 1] - phi[
                                                 N_ghost - 1:-N_ghost - 1]) / (
                       2 * np.diff(x))
    return flux


def first_order_upwind(x, phi, v, N_ghost, alpha):
    flux = -v * (phi[N_ghost:-N_ghost] - phi[
                                         N_ghost - 1:-N_ghost - 1]) / np.diff(x)
    return flux


def third_order_upwind(x, phi, v, N_ghost, alpha):
    flux = -v * (
            phi[N_ghost:-N_ghost] / 2.0 + phi[N_ghost + 1:-N_ghost + 1] / 3.0
            - phi[N_ghost - 1:-N_ghost - 1]
            + phi[N_ghost - 2:-N_ghost - 2] / 6.0) / np.diff(x)
    return flux


def MC(x, phi, v, N_ghost, alpha):
    """MC limiter scheme"""
    phi_jp1_MP = phi[N_ghost:-N_ghost] \
                 + minmod(phi[N_ghost + 1:-N_ghost + 1] - phi[N_ghost:-N_ghost],
                          alpha * (phi[N_ghost:-N_ghost]
                                   - phi[N_ghost - 1:-N_ghost - 1]))

    phi_jm1_MP = phi[N_ghost - 1:-N_ghost - 1] \
                 + minmod(phi[N_ghost:-N_ghost] - phi[N_ghost - 1:-N_ghost - 1],
                          alpha * (phi[N_ghost - 1:-N_ghost - 1]
                                   - phi[N_ghost + 2:-N_ghost + 2]))

    phi_jp1_3u = (5.0 / 6.0) * phi[N_ghost:-N_ghost] \
                 + (1.0 / 3.0) * phi[N_ghost + 1:-N_ghost + 1] \
                 - (1.0 / 6.0) * phi[N_ghost - 1:-N_ghost - 1]

    phi_jm1_3u = (5.0 / 6.0) * phi[N_ghost - 1:-N_ghost - 1] \
                 + (1.0 / 3.0) * phi[N_ghost:-N_ghost] \
                 - (1.0 / 6.0) * phi[N_ghost - 2:-N_ghost - 2]

    phi_jphalf_MC = phi[N_ghost:-N_ghost] \
                    + minmod(phi_jp1_3u - phi[N_ghost:-N_ghost],
                             phi_jp1_MP - phi[N_ghost:-N_ghost])
    phi_jmhalf_MC = phi[N_ghost - 1:-N_ghost - 1] \
                    + minmod(phi_jm1_3u - phi[N_ghost - 1:-N_ghost - 1],
                             phi_jm1_MP - phi[N_ghost - 1:-N_ghost - 1])

    flux = -v * (phi_jphalf_MC - phi_jmhalf_MC) / np.diff(x)

    return flux


# Suresh-Hyunh scheme
def MP5(x, phi_master, v, N_ghost, alpha):
    """
    This method computes the MP5 reconstruction, also called Suresh-Hyunh
    scheme.
    """
    phi_j_final = np.zeros([2, len(x)-1])

    # j = 2 is the j-half case and j = 3 is the j+half case
    for j in [2, 3]:
        # This is the MP5 original reconstruction
        phi_jph_orig = (2.0 * phi_master[j - 2]
                        - 13.0 * phi_master[j - 1]
                        + 47.0 * phi_master[j]
                        + 27 * phi_master[j + 1]
                        - 3.0 * phi_master[j + 2]) / 60.0  # for right-going wind

        phi_MP = phi_master[j] \
                 + minmod(np.array([phi_master[j + 1] - phi_master[j],
                                    alpha * (phi_master[j]
                                             - phi_master[j - 1])]))

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

    # the flux without Reimann solver. Not considering waves in both direction
    flux = -v * (phi_j_final[1, :] - phi_j_final[0,
                                     :]) / dx
