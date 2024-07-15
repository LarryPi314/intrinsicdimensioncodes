import numpy as np

from create_new_scheme_points import create_new_scheme_points
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel
from vonMises import sample_vmf

from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def generate_X(disttype, n, d):
    """
    Generate data points based on the specified distribution type.

    Parameters:
    disttype (str): Distribution type for data points. Must be 'uniform' or 'Gaussian'.
    n (int): Number of data points.
    d (int): Dimensionality of data points.

    Returns:
    ndarray: The generated data points.
    """
    if disttype == 'uniform':
        X = np.random.rand(n, d)
    elif disttype == 'Gaussian':
        X = np.random.randn(n, d)
    elif disttype == 'sphere':
        X = np.random.randn(n, 3)
        X /= np.linalg.norm(X, axis=1)[:, None]
    elif disttype == 'swiss-roll':
        X, t = make_swiss_roll(n_samples=n, noise=0.05, random_state=0)
    elif disttype == 'vonMises':
        mu = np.array([-np.sqrt(1/5), -np.sqrt(1/5), -np.sqrt(1/5), -np.sqrt(1/5), -np.sqrt(1/5), 0])
        kappa = 10
        X = sample_vmf(mu, kappa, n)
    else:
        raise ValueError("disttype must be 'uniform' or 'Gaussian'")

    return X
def average_repmat(reps, disttype, kerneltype, total_pt_num, select_pt_num, d, l):
    """
    Compute the average of replicated singular values for a given number of iterations.

    Parameters:
    reps (int): Number of iterations to average eigenvalues over. 
    disttype (str): Type of distribution for generating points. Must be either 'uniform' or 'Gaussian'.
    kerneltype (str): Type of kernel. Must be either 'Gaussian' or 'Cauchy'.
    total_pt_num (int): Total number of points (vectors randomly generated).
    select_pt_num (int): Number of points to select (among those vectors randomly generated).
    d (int): Dimension of the points (vectors).
    l (float): Bandwidth parameter for the kernel matrix.

    Returns:
    numpy.ndarray: Array of average values for each point.

    Raises:
    ValueError: If disttype is not 'uniform' or 'Gaussian'.
    ValueError: If kerneltype is not 'Gaussian' or 'Cauchy'.
    """
    tempresult = []

    if kerneltype == 'Gaussian':
        cnt = 0
        for j in range(reps):
            X = generate_X(disttype, total_pt_num, d)
            Xk = create_new_scheme_points(X, total_pt_num, select_pt_num, d)
            B = generate_gaussian_kernel(Xk, l)
            svd_vals = np.linalg.svd(B, compute_uv=False)
            
            cnt += 1
            replicated_svd_vals = np.tile(svd_vals, (total_pt_num // select_pt_num, 1)).flatten()
            if cnt == 1:
                print(svd_vals)
                print(len(svd_vals))
                print(replicated_svd_vals)
                print(len(replicated_svd_vals))
            tempresult.append(np.sort(replicated_svd_vals)[::-1])

    elif kerneltype == 'Cauchy':
        for j in range(reps): 
            X = generate_X(disttype, total_pt_num, d)
            Xk = create_new_scheme_points(X, total_pt_num, select_pt_num, d)
            B = generate_cauchy_kernel(Xk, l)
            svd_vals = np.linalg.svd(B, compute_uv=False)
            replicated_svd_vals = np.tile(svd_vals, (total_pt_num // select_pt_num, 1)).flatten()
            tempresult.append(np.sort(replicated_svd_vals)[::-1])
    else:
        raise ValueError("kerneltype must be 'Gaussian' or 'Cauchy'")
    
    # Compute the average of the replicated singular values
    result = np.zeros(total_pt_num)
    print(total_pt_num)
    print(len(tempresult[0]))
    print(reps)

    result_length = len(tempresult[0])
    result = np.zeros(result_length)
    for k in range(result_length):
        avglist = [tempresult[j][k] for j in range(reps)]
        result[k] = np.mean(avglist)

    return result

# uncomment to visualize datasets
# def make_sphere(total_pt_num, d):
#     '''
#     Samples points from a uniform spherical distribution using numpy
#     '''
#     X = np.random.randn(total_pt_num, d)
#     X /= np.linalg.norm(X, axis=1)[:, None]
#     return X

# if __name__ == "__main__":
#     from sklearn.datasets import make_swiss_roll
#     X, t = make_swiss_roll(n_samples=500,noise=0.05, random_state=0)
#     # X = make_sphere(500, 3)
#     # plot the swiss roll
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.viridis)
#     plt.show()