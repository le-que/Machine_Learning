import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        exp_max = np.exp(logit - np.max(logit,axis=-1,keepdims=True))
        return exp_max/np.sum(exp_max,axis=-1,keepdims=True)

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        maxval = np.max(logit, axis=1, keepdims=True)
        logit = logit - maxval
        return np.log(np.sum(np.exp(logit), axis=1, keepdims=True)) + maxval
    
    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        sigma = np.diagonal(sigma_i)
        x = np.exp(-np.square(points - mu_i[np.newaxis,:])/(2*sigma))
        y = x/np.sqrt(2*np.pi*sigma)[np.newaxis,:]
        return np.prod(y,axis = 1)

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """
        N, D = points.shape
        try:
            inv = np.linalg.inv(sigma_i)
        except np.linalg.LinAlgError:
            inv = np.linalg.inv(sigma_i + SIGMA_CONST * np.identity(D))
        norm_factor = np.sqrt((2*np.pi)**D * np.abs(np.linalg.det(sigma_i)))
        npdf = np.exp(-np.sum(np.dot(points - mu_i, inv) * (points - mu_i), axis=1) / 2) / norm_factor
        return npdf


    def create_pi(self):
        """
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        return np.ones(self.K) * (1.0/self.K)

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        return np.random.uniform(low=42.0, high=95.0, size=self.K)
    
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        return np.eye(len(self.points[0])) * self.k
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        N, D = self.points.shape
        pi = np.full(shape=self.K, fill_value = 1/self.K)
        mu = self.points[np.random.choice(self.N, self.K, replace=False), :]
        sigma = np.zeros((self.K, D, D))
        for k in range(self.K):
            sigma[k, :, :] = np.eye(D)
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        if full_matrix is False:
            ll = np.zeros((self.points.shape[0], len(mu)))
            for j in range(len(mu)):
                ll[:, j] = np.log(pi[j]+LOG_CONST) + np.log(self.normalPDF(self.points, mu[j], sigma[j])+LOG_CONST)
        if full_matrix is True:
            ll= np.zeros((self.N, self.K))
            for i in range(self.K):
                    ll[:, i] = np.log(pi[i] + LOG_CONST) + np.log(self.multinormalPDF(self.points, mu[i], sigma[i]) + LOG_CONST)
        return ll
    
    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        if full_matrix is False:
            return self.softmax(self._ll_joint(pi, mu, sigma))
        elif full_matrix is True:
            return self.softmax(self._ll_joint(pi, mu, sigma, full_matrix))

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        N, D = self.points.shape
        if full_matrix is True:
            K = gamma.shape[1]
            mu = np.zeros((K, D))
            sigma = np.zeros((K, D, D))
            pi = np.zeros(K)
            for i in range(K):
                gamma_i = gamma[:, i]
                pi[i] = np.sum(gamma_i) / N
                mu[i, :] = np.sum(self.points * gamma_i.reshape(N, 1), axis=0)/np.sum(gamma_i)
                diff = self.points - mu[i, :]
                sigma[i, :, :] = np.dot(gamma_i * diff.T, diff)/np.sum(gamma_i)
                sigma[i, :, :] += np.eye(D)*SIGMA_CONST
        else:
            pi = gamma.sum(axis=0)/len(self.points)
            mu = np.dot(gamma.T, self.points)/gamma.sum(axis = 0)[:,np.newaxis]
            sigma = np.zeros((len(mu), D, D))
            for k in range(np.shape(gamma)[1]):
                x = self.points - mu[k]
                w = gamma[:,k].T * x.T
                y = np.diagonal(np.dot(w, x)/gamma[:,k].sum(0))
                for d in range(D):
                    sigma[k,d,d] = y[d]
        return pi, mu, sigma
        

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

