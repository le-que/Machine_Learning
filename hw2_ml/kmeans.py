
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        return self.points[np.random.choice(self.points.shape[0], self.K, replace=False)]

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        center_indices = np.random.choice(self.points.shape[0], int(0.01 * self.points.shape[0]), replace= False)
        sample_dataset = self.points[center_indices]
        cluster_center_index = np.random.choice(center_indices, 1, replace= False)
        k_centers = self.points[cluster_center_index]
        for i in range(0, self.K - 1):
            dist_matrix  = pairwise_dist(sample_dataset, k_centers)
            dist_matrix = np.min(dist_matrix, axis = 1)
            max_dist_index = np.argmax(dist_matrix)
            k_centers = np.vstack([k_centers, sample_dataset[max_dist_index]])
        self.centers = k_centers
        return self.centers

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison. 
        """        

        self.assignments = np.argmin(pairwise_dist(self.points,self.centers),axis=1)
        return self.assignments

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        centers = np.zeros(self.centers.shape)
        for i in range(len(self.centers)):
            centers[i] = np.mean(self.points[self.assignments == i], axis=0)
            self.centers = centers
        return self.centers

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        self.loss = np.sum(np.square((self.points - self.centers[self.assignments])))
        return self.loss

    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                - The convergence criteria is measured by whether the percentage difference
                    in loss compared to the previous iteration is less than the given 
                    relative tolerance threshold (self.rel_tol). 
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.   
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        for i in range(self.max_iters):
            self.update_assignment()
            new_centers = self.update_centers()
            empty_clusters = np.where(np.isnan(new_centers).any(axis=1))[0]
            if len(empty_clusters) > 0:
                # Handle empty clusters by choosing a random point as the new center
                for cluster in empty_clusters:
                    random_point = np.random.choice(len(self.points))
                    new_centers[cluster] = self.points[random_point]
            self.centers = new_centers
            new_loss = self.get_loss()
            if i > 0 and abs(new_loss - self.loss) / abs(self.loss) < self.rel_tol:
                break
            self.loss = new_loss
        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        N = x.shape[0]
        M = y.shape[0]
        dot_x = (x*x).sum(axis = 1).reshape((N,1))*np.ones(shape=(1,M))
        dot_y = (y*y).sum(axis = 1) * np.ones(shape = (N,1))
        d_squared = dot_x+dot_y-2*x.dot(y.T)
        d_squared[np.less(d_squared, 0.0)] = 0.0
        return np.sqrt(d_squared)

def rand_statistic(xGroundTruth, xPredicted): # [5 pts]
    """
    Args:
        xPredicted : N x 1 numpy array, N = no. of test samples
        xGroundTruth: N x 1 numpy array, N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float
    """
    assert len(xGroundTruth) == len(xPredicted), "Input arrays must have the same length."

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(len(xGroundTruth)):
        for j in range(i+1, len(xGroundTruth)):
            same_ground_truth = (xGroundTruth[i] == xGroundTruth[j])
            same_predicted = (xPredicted[i] == xPredicted[j])
            if same_ground_truth and same_predicted:
                TP += 1
            elif not same_ground_truth and not same_predicted:
                TN += 1
            elif same_ground_truth and not same_predicted:
                FN += 1
            else:
                FP += 1

    rand_index = (TP + TN) / (TP + TN + FP + FN)
    return rand_index