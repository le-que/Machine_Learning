import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        cluster_idx = np.zeros(self.dataset.shape[0], dtype = np.intp) - 1
        visitedIndices = set()
        c = 0
        for idx, point in enumerate(self.dataset):
            if idx not in visitedIndices:
                neighborIndices = self.regionQuery(idx)
                if len(neighborIndices) < self.minPts:
                    cluster_idx[idx] = -1
                else:
                    cluster_idx[idx] = c
                    self.expandCluster(idx, neighborIndices, c, cluster_idx, visitedIndices)
                    c += 1
        return cluster_idx

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints:  
            1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
            2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
        """
        i = 0
        while i < len(neighborIndices):
            j = neighborIndices[i]
            if j not in visitedIndices:
                visitedIndices.add(j)
                neighbors = self.regionQuery(j)
                if len(neighbors) >= self.minPts:
                    neighborIndices = np.concatenate((neighborIndices, neighbors))
            if cluster_idx[j] == -1:
                cluster_idx[j] = C
            i += 1
        return

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        dists = pairwise_dist(self.dataset[pointIndex:pointIndex+1], self.dataset)
        indices = np.argwhere(dists[0] <= self.eps).flatten()
        return indices
        