"""

This script contains main driver class for the shockwave modelling.

:copyright:
    Srijan B. Das (sbdas@princeton.edu)
    Lucas Sawade (lsawade@princeton.edu)

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""




import numpy
import matplotlib


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
                 alpha=4.0):
        """
        This method handles initialization of the problem
        :param dim:
        :param cells:
        :param time_ep_method:
        :param CFL:
        :param dx:
        :param sigma:
        :param v:
        :param T:
        :param alpha:
        """

        self.dim = dim
        self.cells = cells
        self.CFL = CFL
        self.dx = dx
        self.sigma = sigma
        self.v = v
        self.alpha = alpha



class FLASHM:
    """Handles running the main program."""

    def __init__(self, config, method="", time_ep_method="simple"):
        """Handles initialization of the main solver"""
