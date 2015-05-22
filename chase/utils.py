import numpy as np

def pfix(p):
    return np.min([np.max([p, 1e-10]), 1-(1e-10)])

