import numpy as np

from create_new_scheme_points import create_new_scheme_points
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

import numpy as np

def average_repmat(l, disttype, kerneltype, total_pt_num, select_pt_num, d):
    """
    Compute the average of replicated singular values for a given number of iterations.

    Parameters:
    l (int): Number of iterations.
    disttype (str): Type of distribution for generating points. Must be either 'uniform' or 'Gaussian'.
    kerneltype (str): Type of kernel. Must be either 'Gaussian' or 'Cauchy'.
    total_pt_num (int): Total number of points (vectors randomly generated).
    select_pt_num (int): Number of points to select (among those vectors randomly generated).
    d (int): Dimension of the points (vectors).

    Returns:
    numpy.ndarray: Array of average values for each point.

    Raises:
    ValueError: If disttype is not 'uniform' or 'Gaussian'.
    ValueError: If kerneltype is not 'Gaussian' or 'Cauchy'.
    """

    if disttype == 'uniform':
        X = np.random.rand(total_pt_num, d)
    elif disttype == 'Gaussian':
        X = np.random.randn(total_pt_num, d)
    else:
        raise ValueError("disttype must be 'uniform' or 'Gaussian'")

    Xk = create_new_scheme_points(X, total_pt_num, select_pt_num, d)

    tempresult = []

    if kerneltype == 'Gaussian':
        for j in range(l):
            B = generate_gaussian_kernel(Xk, 1/500)
            svd_vals = np.linalg.svd(B, compute_uv=False)
            replicated_svd_vals = np.tile(svd_vals, (total_pt_num // select_pt_num, 1)).flatten()
            tempresult.append(np.sort(replicated_svd_vals)[::-1])

    elif kerneltype == 'Cauchy':
        for j in range(l):
            B = generate_cauchy_kernel(Xk, 10000)
            svd_vals = np.linalg.svd(B, compute_uv=False)
            replicated_svd_vals = np.tile(svd_vals, (total_pt_num // select_pt_num, 1)).flatten()
            tempresult.append(np.sort(replicated_svd_vals)[::-1])
    else:
        raise ValueError("kerneltype must be 'Gaussian' or 'Cauchy'")
    
    # Compute the average of the replicated singular values
    result = np.zeros(total_pt_num)
    for k in range(total_pt_num):
        avglist = [tempresult[j][k] for j in range(l)]
        result[k] = np.mean(avglist)

    return result

