import numpy as np

def generate_gaussian_kernel(vects, l):
    """
    Generate a Gaussian kernel matrix based on the pairwise squared distances between data points.

    Parameters:
    X (ndarray): The input data points.
    sigma (float): The standard deviation parameter for the Gaussian kernel.

    Returns:
    ndarray: The Gaussian kernel matrix.
    """
    n = vects.shape[0]
    A = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            dist_sq = np.sum((vects[j, :] - vects[k, :]) ** 2)
            A[j, k] = np.exp(-dist_sq / l) 
    return A

