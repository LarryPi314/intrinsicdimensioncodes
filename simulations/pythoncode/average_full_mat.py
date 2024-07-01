import numpy as np

from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

from sklearn.datasets import make_swiss_roll

def average_full_mat(s, disttype, kerneltype, n, d, l):
    """
    Generate an average kernel matrix based on the specified distribution type and kernel type.
    This function obtains the average eigenvalues of the kernel matrices generated
    using the specified distribution type and kernel type.

    Parameters:
    l (int): Number of kernel matrices to generate.
    disttype (str): Distribution type for data points. Must be 'uniform' or 'Gaussian'.
    kerneltype (str): Kernel type. Must be 'Gaussian' or 'Cauchy'.
    n (int): Number of data points.
    d (int): Dimensionality of data points.

    Returns:
    ndarray: The average kernel matrix.
    """
    if disttype == 'uniform':
        X = np.random.rand(n, 3)
    elif disttype == 'Gaussian':
        X = np.random.randn(n, d)
    elif disttype == 'sphere':
        X = np.random.randn(n, 4)
        X /= np.linalg.norm(X, axis=1)[:, None]
    elif disttype == 'swiss-roll':
        X, t = make_swiss_roll(n_samples=n, noise=0.05, random_state=0)
    else:
        raise ValueError("disttype must be 'uniform' or 'Gaussian'")

    tempresult = []

    if kerneltype == 'Gaussian':
        for j in range(s):
            A = generate_gaussian_kernel(X, l) 
            tempresult.append(np.sort(np.linalg.svd(A, compute_uv=False))[::-1])
    elif kerneltype == 'Cauchy':
        for j in range(s):
            A = generate_cauchy_kernel(X, l)
            tempresult.append(np.sort(np.linalg.svd(A, compute_uv=False))[::-1])
    else:
        raise ValueError("kerneltype must be 'Gaussian' or 'Cauchy'")

    result_full = np.zeros(n)
    for k in range(n):
        avglist = [tempresult[j][k] for j in range(s)]
        result_full[k] = np.mean(avglist)

    return result_full
