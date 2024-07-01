import numpy as np

from vonMises import sample_vmf
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

from sklearn.datasets import make_swiss_roll

def generate_X(disttype, n, d):
    """
    Generate data points based on the specified distribution type.

    Parameters:
    disttype (str): Distribution type for data points. Must be 'uniform' or 'Gaussian'.
    n (int): Number of data points.
    d (int): Dimensionality of data points.

    Returns:
    ndarray: The average kernel matrix.
    """
    mu = np.array([-np.sqrt(1/5), -np.sqrt(1/5), -np.sqrt(1/5), -np.sqrt(1/5), -np.sqrt(1/5), 0])
    kappa = 10
    if disttype == 'uniform':
        X = np.random.rand(n, 5)
    elif disttype == 'Gaussian':
        X = np.random.randn(n, 2)
    elif disttype == 'sphere':
        X = np.random.randn(n, 5)
        X /= np.linalg.norm(X, axis=1)[:, None]
    elif disttype == 'swiss-roll':
        X, t = make_swiss_roll(n_samples=n, noise=0.05, random_state=0)
    elif disttype == 'vonMises':
        X = sample_vmf(mu, kappa, n)
    else:
        raise ValueError("invalid distribution")
    return X

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
    tempresult = []

    if kerneltype == 'Gaussian':
        X = generate_X(disttype, n, d)
        for j in range(l):
            A = generate_gaussian_kernel(X, 1/100) # play around with sigma
            tempresult.append(np.sort(np.linalg.svd(A, compute_uv=False))[::-1])
    elif kerneltype == 'Cauchy':
        X = generate_X(disttype, n, d)
        for j in range(l):
            A = generate_cauchy_kernel(X, 10000)
            tempresult.append(np.sort(np.linalg.svd(A, compute_uv=False))[::-1])
    else:
        raise ValueError("kerneltype must be 'Gaussian' or 'Cauchy'")

    result_full = np.zeros(n)
    for k in range(n):
        avglist = [tempresult[j][k] for j in range(l)]
        result_full[k] = np.mean(avglist)

    return result_full
