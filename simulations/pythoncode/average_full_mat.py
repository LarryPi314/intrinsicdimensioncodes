import numpy as np

from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

def average_full_mat(l, disttype, kerneltype, n, d):
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
        X = np.random.rand(n, d)
    elif disttype == 'Gaussian':
        X = np.random.randn(n, d)
    else:
        raise ValueError("disttype must be 'uniform' or 'Gaussian'")

    tempresult = []

    if kerneltype == 'Gaussian':
        for j in range(l):
            A = generate_gaussian_kernel(X, 1/500)
            tempresult.append(np.sort(np.linalg.svd(A, compute_uv=False))[::-1])
    elif kerneltype == 'Cauchy':
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
