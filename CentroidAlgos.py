import numpy as np

from random import choice
from .common import eucl_dist


class CentroidAlgos():
    MAX_ITER = 300


    def kmeans_with_auto_elbow(self, ata):
        #

    
    def kmeans(self, data, k):
        if k == -1:
            k = kmeans_with_auto_elbow(data)

        centroids = kmeans_init_centroids(data, k)
        prev_centroids = np.empty(shape=[0, data.shape[1]])
        cluster_assignment = []

        n_iter = 0
        while not kmeans_terminate(prev_centroids, centroids, n_iter):
            n_iter += 1

            # Assignment step: assign each data point to the closest centroid
            for i in range(len(data)):
                distances = eucl_dist(data[i], centroids)
                cluster_assignment[i] = np.argmin(distances)

            prev_centroids = np.copy(centroids)

            # Centroid update step: update centroids via the average value of points assigned to it
            for i in range(k):
                assigned_points = [data[j] for j in range(len(data)) if cluster[j] == i]
                if assigned_points:
                    centroids[i] = np.mean(assigned_points, axis=0)

        return cluster_assignment


    def kmeans_plus_plus(self, data, k):
        if k == -1:
            k = kmeans_with_auto_elbow(data)

        centroids = kmeans_pp_init_centroids(data, k)
        prev_centroids = np.empty(shape=[0, data.shape[1]])
        cluster_assignment = []

        n_iter = 0
        while not kmeans_terminate(prev_centroids, centroids, n_iter):
            n_iter += 1

            # Assignment step: assign each data point to the closest centroid
            for i in range(len(data)):
                distances = eucl_dist(data[i], centroids)
                cluster_assignment[i] = np.argmin(distances)
            
            prev_centroids = np.copy(centroids)

            # Centroid update step: update centroids via the average value of points assigned to it
            for i in range(k):
                assigned_points = [data[j] for j in range(len(data)) if cluster[j] == i]
                if assigned_points:
                    centroids[i] = np.mean(assigned_points, axis=0)
        
        return cluster_assignment


    def kmeans_bisecting(self, data, k, n_trials):
        if k == -1:
            k = kmeans_with_auto_elbow(data)

        centroids, cluster_assigment = kmeans_bisect_init_centroids(self, data)
        prev_centroids = np.empty(shape=[0, data.shape[1]])


    def kmeans_terminate(self, prev_centroids, curr_centroids, curr_iter):
        if curr_iter > MAX_ITER:
            return True
        return np.array_equal(prev_centroids, curr_centroids)


    def kmeans_init_centroids(self, data, k, rnd_num=56):
        np.random.seed(rnd_num)
        centroids = []
        m = np.shape(data)[0]
        for _ in range(k):
            centroid = np.random.randint(0, m-1)
            centroids.append(centroid)
        return np.array(centroids)
(

    def kmeans_pp_init_centroids(self, data, k):
        centroids = []
        centroids.append(choice(data))
        for _ in range(k - 1):
            sq_dists = np.array([min([np.square(eucl_dist(x, c)) for c in centroids]) for x in data])
            probs = sq_dists / sq_dists.sum()
            cum_probs = probs.cumsum()
            rnd = np.random.random()
            selected_cluster = np.where(cum_probs >= rnd)[0][0]
            centroids.append(selected_cluster)
        return np.array(centroids)
    

    def kmeans_bisect_init_centroids(self, data):
        centroids = []
        centroids[0] = np.mean(data, axis=0)
        cluster_assignment = np.full(data.shape, centroids[0])
        return np.array(centroids), cluster_assignment


