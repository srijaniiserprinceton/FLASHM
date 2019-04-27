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


def minmod_two(x, y):
    return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))


def minmod(x_arr):
    x_array = np.array(x_arr)
    x_twoelement_sum = 0.5 * (np.sign(x_array[0, :]) + np.sign(x_array[1:, :]))
    x_twoelement_sum[1:, :] = np.abs(x_twoelement_sum[1:, :])

    sign_minmod = np.prod(x_twoelement_sum, axis=0)
    minmod_val = sign_minmod * np.amin(np.abs(x_array), axis=0)

    return minmod_val


def second_order_centered(x, phi, v, N_ghost, s, alpha):
    """ Computes the flux using the 2nd order centered scheme.
    The phi that is taken in is padded. The padding depends on the
    boundary conditions.
    :param s: shift vector
    """
    phi_out = np.zeros([2, len(x)-1])

    phi_out[0, :] = (phi[N_ghost + s[1, 2]:-N_ghost + s[1, 2]]
                     + phi[N_ghost + s[1, 3]:-N_ghost + s[1, 3]]) / 2.0

    phi_out[1, :] = (phi[N_ghost + s[0, 3]:-N_ghost + s[0, 3]]
                     + phi[N_ghost + s[0, 4]:-N_ghost + s[0, 4]]) / 2.0

    return phi_out

def first_order_upwind(x, phi, v, N_ghost, s, alpha):

    phi_out = np.zeros([2, len(x) - 1])

    phi_out[0, :] = phi[N_ghost + s[1, 2]:-N_ghost + s[1, 2]]
    phi_out[1, :] = phi[N_ghost + s[0, 3]:-N_ghost + s[0, 3]]

    return phi_out


def third_order_upwind(x, phi, v, N_ghost, s, alpha):

    N_cell = len(x) - 1
    phi_out = np.zeros([2, N_cell])

    phi_out[0, :] = phi[N_ghost + s[1, 2]:-N_ghost + s[1, 2]] \
                        + 0.25 * (phi[N_ghost + s[1, 5]:-N_ghost + s[1, 5]]
                                  - phi[N_ghost + s[1, 3]:-N_ghost + s[1, 3]]) \
                        + (1.0 / 12.0) \
                            * (phi[N_ghost + s[1, 5]:-N_ghost + s[1, 5]]
                               + phi[N_ghost + s[1, 3]:-N_ghost + s[1, 3]]
                               - 2 * phi[N_ghost + s[1, 4]:-N_ghost +s[1, 4]])

    phi_out[1, :] = phi[N_ghost + s[0, 3]:-N_ghost + s[0, 3]] \
                        + 0.25 * (phi[N_ghost + s[0, 4]:-N_ghost + s[0, 4]]
                                  - phi[N_ghost + s[0, 2]:-N_ghost + s[0, 2]]) \
                        + (1.0 / 12.0) \
                    * (phi[N_ghost + s[0, 4]:-N_ghost + s[0, 4]]
                       + phi[N_ghost + s[0, 2]:-N_ghost + s[0, 2]]
                       - 2 * phi[N_ghost + s[0, 3]:-N_ghost + s[0, 3]])

    return phi_out


def MC(x, phi, v, N_ghost, s, alpha):
    """MC limiter scheme"""
    
    phi_jp1_MP = phi[N_ghost + s[3]:-N_ghost + s[3]] \
                 + minmod_two(phi[N_ghost + s[4]:-N_ghost + s[4]]
                              - phi[N_ghost + s[3]:-N_ghost + s[3]],
                          alpha * (phi[N_ghost + s[3]:-N_ghost + s[3]]
                                   - phi[N_ghost + s[2]:-N_ghost + s[2]]))

    phi_jm1_MP = phi[N_ghost + s[2]:-N_ghost + s[2]] \
                 + minmod_two(phi[N_ghost + s[3]:-N_ghost + s[3]]
                              - phi[N_ghost + s[2]:-N_ghost + s[2]],
                          alpha * (phi[N_ghost + s[2]:-N_ghost + s[2]]
                                   - phi[N_ghost + s[5]:-N_ghost + s[5]]))

    phi_jp1_3u = (5.0 / 6.0) * phi[N_ghost + s[3]:-N_ghost + s[3]] \
                 + (1.0 / 3.0) * phi[N_ghost + s[4]:-N_ghost + s[4]] \
                 - (1.0 / 6.0) * phi[N_ghost + s[2]:-N_ghost + s[2]]

    phi_jm1_3u = (5.0 / 6.0) * phi[N_ghost + s[2]:-N_ghost + s[2]] \
                 + (1.0 / 3.0) * phi[N_ghost + s[3]:-N_ghost + s[3]] \
                 - (1.0 / 6.0) * phi[N_ghost + s[1]:-N_ghost + s[1]]

    phi_jphalf_MC = phi[N_ghost + s[3]:-N_ghost + s[3]] \
                    + minmod_two(phi_jp1_3u
                                 - phi[N_ghost + s[3]:-N_ghost + s[3]],
                             phi_jp1_MP - phi[N_ghost + s[3]:-N_ghost + s[3]])

    phi_jmhalf_MC = phi[N_ghost + s[2]:-N_ghost + s[2]] \
                    + minmod_two(phi_jm1_3u
                                 - phi[N_ghost + s[2]:-N_ghost + s[2]],
                             phi_jm1_MP - phi[N_ghost + s[2]:-N_ghost + s[2]])

    flux = -v * (phi_jphalf_MC - phi_jmhalf_MC) / np.diff(x)

    return flux


def MP5(x, phi, v, N_ghost, s, alpha):
    """
    This method computes the MP5 reconstruction, also called Suresh-Hyunh
    scheme.
    """
    # Set epsilon Shock
    eps_SH = 1e-10

    # Initialize
    phi_j_final = np.zeros([2, len(x)-1])
    
    for j in [0, 1]:  # j = 0 is the j-half case and j = 1 is the j+half case
        # This is the MP5 original reconstruction

        # for right-going wind
        phi_jph_orig = (2.0 * phi[N_ghost + s[0] + j:-N_ghost + s[0] + j] 
                        - 13.0 * phi[N_ghost + s[1] + j:-N_ghost + s[1] + j] 
                        + 47.0 * phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] 
                        + 27 * phi[N_ghost + s[3] + j:-N_ghost + s[3] + j] - 3.0 *
                        phi[N_ghost + s[4] + j:-N_ghost + s[4] + j]) / 60.0

        phi_MP = phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] \
                    + minmod(np.array([phi[N_ghost + s[3] + j:-N_ghost + s[3] + j]
                                       - phi[N_ghost + s[2] + j:-N_ghost + s[2] + j],
                                       alpha
                                       * (phi[N_ghost + s[2] + j:-N_ghost + s[2] + j]
                                          - phi[N_ghost - 2 + j:-N_ghost - 2
                                                                + j])]))

        # creating the array to be compared with epsilon at each interface
        cond_array = np.multiply(phi_jph_orig
                                 - phi[N_ghost + s[2] + j:-N_ghost + s[2] + j],
                                 phi_jph_orig - phi_MP)

        # performing simple assignment if the condition is satisfied
        phi_j_final[j - 2, cond_array <= eps_SH] = phi_jph_orig[
            cond_array <= eps_SH]

        # further computation if the condition is not satisfied
        d = phi[N_ghost + s[1] + j:-N_ghost + s[1] + j] \
            + phi[N_ghost + s[3] + j:-N_ghost + s[3] + j] \
            - 2 * phi[N_ghost + s[2] + j:-N_ghost + s[2] + j]
        
        dp1 = phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] \
              + phi[N_ghost + s[4] + j:-N_ghost + s[4] + j] \
              - 2 * phi[N_ghost + s[3] + j:-N_ghost + s[3] + j]

        dm1 = phi[N_ghost + s[0] + j:-N_ghost + s[0] + j] \
              + phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] \
              - 2 * phi[N_ghost + s[1] + j:-N_ghost + s[1] + j]

        d_M4_jph = minmod(np.array([4.0 * d - dp1, 4.0 * dp1 - d, d, dp1]))
        d_M4_jmh = minmod(np.array([4.0 * dm1 - d, 4.0 * d - dm1, dm1, d]))

        phi_UL = phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] \
                 + alpha * (phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] 
                            - phi[N_ghost + s[1] + j:-N_ghost + s[1] + j])

        phi_AV = 0.5 * (phi[N_ghost + s[2] + j:-N_ghost + s[2] + j]
                        + phi[N_ghost + s[3] + j:-N_ghost + s[3] + j])

        phi_MD = phi_AV - 0.5 * d_M4_jph

        phi_LC = phi[N_ghost + s[2] + j:-N_ghost + s[2] + j] \
                 + 0.5 * (phi[N_ghost + s[2] + j:-N_ghost + s[2] + j]
                              - phi[N_ghost + s[1] + j:-N_ghost + s[1] + j]) \
                 + (4.0 / 3.0) * d_M4_jmh

        phi_min = np.maximum(
            np.minimum(phi[N_ghost + s[2] + j:-N_ghost + s[2] + j],
                       phi[N_ghost + s[3] + j:-N_ghost + s[3] + j], phi_MD),
            np.minimum(phi[N_ghost + s[2] + j:-N_ghost + s[2] + j],phi_UL,
                       phi_LC))
        
        phi_max = np.minimum(np.maximum(phi[N_ghost + s[2] + j:-N_ghost + s[2] + j], 
                                        phi[N_ghost + s[3] + j:-N_ghost + s[3] + j], phi_MD),
                             np.maximum(phi[N_ghost + s[2] + j:-N_ghost + s[2] + j],
                                        phi_UL,
                                        phi_LC))

        # for the interfaces where the condition array did not satisfy the condition
        phi_j_final[j - 2, cond_array > eps_SH] = (phi_jph_orig + minmod(
            np.array([phi_min - phi_jph_orig, phi_max - phi_jph_orig])))[
            cond_array > eps_SH]

    flux = -v * (phi_j_final[1, :] - phi_j_final[0, :]) / np.diff(x)

    return flux