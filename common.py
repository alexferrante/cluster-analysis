import math
import numpy as np


def euclidean_distance(p, q, axis=1):
    return np.sqrt(np.sum(np.square(p - q)))


def manhattan_distance(p, q):
    return np.sum(np.fabs(p-q))


def minkowski_distance(p, q, power):
    return np.power(np.sum(np.power(np.fabs(p-q), power)), 1/power)


def chebyshev_distance(p, q):
    return np.max(np.fabs(p-q))


def cosine_similarity(p, q):
    num = float(np.dot(p, q.T))
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    return num / denom



class OutlierDetection:
    #