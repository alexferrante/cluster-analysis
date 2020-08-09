import numpy as np

from random import choice
from .common import eucl_dist


class CentroidAlgos():
    MAX_ITER = 300


    def kmeans_with_auto_elbow(data):
        #


    def kmeans(data, k, centroid_opt):
        if k == -1:
            k = kmeans_with_auto_elbow(data)
        if centroid_opt == "rnd":
            centroids = kmeans_rnd_centroids(data, k)
        elif centroid_opt == "++":
            centroids = kmeans_plus_centroids(data, k)
        n_iter = 0
        prev_centroids = np.empty(shape=[0, data.shape[1]])
        while not kmeans_terminate(prev_centroids, centroids, n_iter):
            n_iter += 1
            for i in range(len(data)):
                distances = eucl_dist(data[i], centroids)
                cluster[i] = np.argmin(distances)
            prev_centroids = np.copy(centroids)
            for i in range(k):
                points = [data[j] for j in range(len(data)) if cluster[j] == i]
                if points:
                    centroid[i] = np.mean(points, axis=0)
        return cluster


    def kmeans_terminate(prev_centroids, curr_centroids, curr_iter):
        if curr_iter > MAX_ITER:
            return True
        return np.array_equal(prev_centroids, curr_centroids)


    def kmeans_rnd_centroids(data, k, rnd_num=56):
        np.random.seed(rnd_num)
        centroids = []
        m = np.shape(data)[0]
        for _ in range(k):
            centroid = np.random.randint(0, m-1)
            centroids.append(centroid)
        return np.array(centroids)


    def kmeans_plus_centroids(data, k, rnd_num=56):
        np.random.seed(rnd_num)
        centroids = []
        centroids.append(choice(data))
        for _ in range(1, k):
            all_sq_dist = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in data])
            prob = all_sq_dist / all_sq_dist.sum()
            cum_prob = prob.cumsum()
            rnd = np.random.rand()
            for cent, p in enumerate(cum_prob):
                if rnd_num < p:
                    i = j
                    break
            centroids.append(data[i])
        return np.array(centroids)