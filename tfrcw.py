import numpy as np

import utils


def tfrcw(**kwargs):
    if len(kwargs) == 0:
        raise ValueError("At least 1 parameter required")
    x = kwargs['x']
    xrow = x.shape[0]
    xcol = x.shape[1]
    if xcol == 0 or xcol > 2:
        raise ValueError("X must have one or two columns")
    if len(kwargs) <= 2:
        N = xrow
    elif xrow < 0:
        raise ValueError("N must be greater than zero")
    elif np.power(2, utils.nextpow2(xrow)) != xrow:
        print("For a faster computation, N should be a power of two")
tfrcw(x = np.asarray([[5,6]]))