import numpy as np

def create_nl_feature(X):
    '''
    Create additional features and add it to the dataset.
    
    Returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    return np.column_stack((X[:, 0], X[:, 1], np.exp(-(X[:, 0]**2 + X[:, 1]**2))))