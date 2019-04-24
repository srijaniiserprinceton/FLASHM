FLASHM
------


Installation
============

.. code:: bash

    $ git clone https://github.com/lsawade/FLASHM
    $ cd FLASHM
    $ pip install .


Main classes and running
===========================

The program is being run by two main classes, the configuration class `Config`,
which sets all parameters of the domain, and the driver class `FLASHM`, which
controls factors such as the time extrapolation as well as the Reconstruction
of the flux between cells.

.. code:: python

    # Set Config
    N = N_cells + 1     # N+1 is the number of cell edges.
    CFL = 0.1           # CFL number
    dx = 1.0 / N_cells  # grid spacing
    sig = 0.05          # sigma for the gaussian in the initial function
    v = 1.0             # advection velocity
    T = 1 / v           # Length of domain in code units is 1.0
    alpha = 4.0         # parameter for defining the MC limited

    # Setting Configuration.
    config = Config(dim=1, cells=200, CFL=0.1, dx=1.0, sigma=0.05, v=1.0,
                    alpha=4.0)

    # Create solver class
    time_stepping = "simple"
    reconstruction_method = "linear"

    flashm = FLASHM(config, reconstruction_method=reconstruction_method,
                    time_ep_method=time_stepping, time_length)

    # Run the solver
    flashm.run()


