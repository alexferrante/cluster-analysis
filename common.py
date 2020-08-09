import numpy as np


def eucl_dist(p1, p2, axis=1):
    return np.linalg.norm(p1-p2, axis=axis)
