import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        squared_errors = (pred - label) ** 2
        mean_squared_error = np.mean(squared_errors)
        rmse_value = np.sqrt(mean_squared_error)

        return rmse_value

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x:
                1-dimensional case: (N,) numpy array
                D-dimensional case: (N, D) numpy array
                Here, N is the number of instances and D is the dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        N, D = x.shape  # Get the shape of the input feature matrix
        
        # Initialize the feature matrix with ones (bias term)
        feat = np.ones((N, degree + 1, D) if D > 1 else (N, degree + 1))

    # Create an array with powers of x up to the specified degree
        powers = np.arange(1, degree + 1)
        
         # Reshape powers to have shape (1, degree)
        powers = powers[:, np.newaxis]

        # Compute the polynomial features without using explicit for loops
        if D > 1:
            feat[:, 1:, :] = x[:, np.newaxis, :] ** powers
        else:
            feat[:, 1:] = x[:, np.newaxis] ** powers

        
        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,1+D) numpy array, where N is the number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            weight: (1+D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return np.dot(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        # Calculate the pseudo-inverse of xtrain
        x_pseudo_inv = np.linalg.pinv(xtrain)
        # Calculate the weights by multiplying the pseudo-inverse with ytrain
        weight = np.dot(x_pseudo_inv,ytrain)
        print(weight)
        return weight
        

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weight = np.zeros((xtrain.shape[1], 1))
        
        loss_per_epoch = []
        
        for epoch in range(epochs):
            # Compute predictions
            y_pred = np.dot(xtrain, weight)
            
            # Compute the error
            error = y_pred - ytrain
            
            rmse = self.rmse(y_pred, ytrain)
            loss_per_epoch.append(rmse)
            
            # Compute gradients
            gradient = np.dot(xtrain.T, error) / xtrain.shape[0]
            
            # Update weights using gradient descent
            weight -= learning_rate * gradient
        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        N, D = xtrain.shape
        weights = np.zeros((D, 1))
        loss_per_epoch = []

        for epoch in range(epochs):
            for i in range(N):
                xi = xtrain[i, :].reshape(1, -1)  # Get the current data point
                yi = ytrain[i, :].reshape(1, -1)  # Get the current label

                # Calculate the prediction and error
                prediction = np.dot(xi, weights)
                error = prediction - yi

                # Update the weights using gradient descent
                gradient = np.dot(xi.T, error)
                weights -= learning_rate * gradient

                # Calculate RMSE and store it
                rmse = self.rmse(prediction, ytrain)
                loss_per_epoch.append(rmse)

        return weights, loss_per_epoch

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is
                    number of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        N, D = xtrain.shape

        b=np.identity(D)

        b[0,0]=0

        a=np.transpose(xtrain) @ xtrain + c_lambda*b

        weight=np.linalg.pinv(a) @ np.transpose(xtrain) @ ytrain

        return weight

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - You should avoid applying regularization to the bias term in the gradient update
        """
        raise NotImplementedError
        

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Hints:
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        raise NotImplementedError

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> List[float]:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each kfold

        Args:
            X : (N,1+D) numpy array, where N is the number of instances
                and D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        num_samples = X.shape[0]
        fold_size = num_samples // kfold
        loss_per_fold = []

        # Create an array of indices for splitting
        indices = np.arange(num_samples)

        for i in range(kfold):
            # Calculate the start and end indices for the current fold
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < kfold - 1 else num_samples

            # Use boolean indexing to split data into training and test sets
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate((indices[:start_idx], indices[end_idx:]), axis=0)

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Fit a ridge regression model
            ridge_weights = self.ridge_fit_closed(X_train, y_train, c_lambda)

            # Predict using the model
            y_pred = X_test.dot(ridge_weights)
            rmse = self.rmse(y_pred, y_test)
            loss_per_fold.append(rmse)
        
        return loss_per_fold

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """

        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm

        return best_lambda, best_error, error_list
