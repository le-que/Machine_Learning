import numpy as np
from typing import Tuple


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((3,N,D) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (1 channel for black and white and 3 channels for RGB)
        Image is the matrix X.

        Hint: np.linalg.svd by default returns the transpose of V. We want you to return the transpose of V, not V.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (3,N,N) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
        """
        if X.ndim == 2:
            U, S, V = np.linalg.svd(X)
        else:
            U = np.zeros((3, X.shape[1], X.shape[1]))
            S = np.zeros((3, np.minimum(X.shape[1], X.shape[2])))
            V = np.zeros((3, X.shape[2], X.shape[2]))
            for i in range(3):
                u, s, v = np.linalg.svd(X[i, :, :])
                U[i, :, :] = u
                S[i, :] = s
                V[i, :, :] = v
        return U, S, V

    def compress(
        self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """Compress the SVD factorization by keeping only the first k components

        Args:
            U (np.ndarray): (N,N) numpy array for black and white simages / (3,N,N) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
            k (int): int corresponding to number of components to keep

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                U_compressed: (N, k) numpy array for black and white images / (3, N, k) numpy array for color images
                S_compressed: (k, ) numpy array for black and white images / (3, k) numpy array for color images
                V_compressed: (k, D) numpy array for black and white images / (3, k, D) numpy array for color images
        """
        if len(U.shape) == 2:
            #Compress
            U_compressed = U[:, :k]
            S_compressed = S[:k]
            V_compressed = V[:k, :]
            return tuple([U_compressed,S_compressed,V_compressed])
        #For color image
        else:
            U_compressed = np.zeros((3, U.shape[1], k))
            S_compressed = np.zeros((3, k))
            V_compressed = np.zeros((3, k, V.shape[2]))
            for i in range(3):
                U_compressed[i, :, :] = U[i, :, :k]
                S_compressed[i, :] = S[i, :k]
                V_compressed[i, :,:] = V[i, :k, :]
            return tuple([U_compressed,S_compressed,V_compressed])

    def rebuild_svd(
        self,
        U_compressed: np.ndarray,
        S_compressed: np.ndarray,
        V_compressed: np.ndarray,
    ) -> np.ndarray:  # [4pts]
        """
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.

        Args:
            U_compressed: (N,k) numpy array for black and white images / (3,N,k) numpy array for color images
            S_compressed: (k, ) numpy array for black and white images / (3,k) numpy array for color images
            V_compressed: (k,D) numpy array for black and white images / (3,k,D) numpy array for color images

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (3,N,D) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U_compressed.shape) == 2:
            S = S_compressed[:]
            U = U_compressed[:, :]
            V = V_compressed[:, :]
            aux = np.matmul(U,np.diag(S))
            X_rebuild = np.matmul(aux,V)
        else:
            X_rebuild = np.zeros((3, U_compressed.shape[1], V_compressed.shape[2]))
            for i in range(3):
                S = S_compressed[i, :]
                U = U_compressed[i, :, :]
                V = V_compressed[i, :, :]
                aux = np.matmul(U,np.diag(S))
                X_rebuild[i, :, :] = np.matmul(aux,V)
        return X_rebuild

    def compression_ratio(self, X: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        if len(X.shape) == 2:
            orig = X.shape[0] * X.shape[1]
            compr = k * (1 + X.shape[0] + X.shape[1])
            compression_ratio = compr/orig
        #For color image
        else:
            orig = X.shape[1] * X.shape[2] * X.shape[0]
            compr = k * (1 + X.shape[1] + X.shape[2]) * X.shape[0]
            compression_ratio = compr/orig
        return float(compression_ratio)

    def recovered_variance_proportion(self, S: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (3,min(N,D)) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        if len(S.shape) == 1:
        #Compute recovered variance
            recovered_var = float(np.sum((S**2)[:k]) / np.sum(S**2))
        #For color image
        else:
            recovered_var = []
        #Iterate over the 3 channels
            for i in range(3):
        #Compute recovered variance
                recovered_var.append(float(np.sum((S**2)[i, :k])) / np.sum((S**2)[i, :]))
        return np.array(recovered_var)

    def memory_savings(
        self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS

        Returns the memory required to store the original image X and
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
            U (np.ndarray): (N,N) numpy array for black and white simages / (3,N,N) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]:
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X
        """

        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (
            U_compressed.nbytes + S_compressed.nbytes + V_compressed.nbytes
        )
        savings = original_nbytes - compressed_nbytes

        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int = 3) -> str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f"{round(n, ndigits)} {units[units_idx]}"
