from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Function to estimate the intrinsic dimension at the sample level using k-nearest neighbors
def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]  # Remove self-distance
    dist = dist[:, 0:k]  # Consider only the k nearest neighbors
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])  # Log ratio of distances
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    return d

# Function to estimate the intrinsic dimension over a range of scales
def intrinsic_dim_scale_interval(X, k1=5, k2=10):
    X = pd.DataFrame(X).drop_duplicates().values  # Remove duplicates
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

# Function to perform repeated intrinsic dimension estimation using bootstrap
def repeated(func, X, nb_iter=100, random_state=None, verbose=0, mode='bootstrap', **func_kw):
    rng = np.random if random_state is None else np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)
    for _ in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError(f'unknown mode: {mode}')
        results.append(func(Xr, **func_kw))
    return results 





def sample_vmf(mu, kappa, num_samples):
    dim = len(mu)
    result = np.zeros((num_samples, dim))
    w = np.zeros(num_samples)

    b = (-2 * kappa + np.sqrt(4 * kappa**2 + (dim - 1)**2)) / (dim - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (dim - 1) * np.log(1 - x0**2)

    for i in range(num_samples):
        while True:
            z = np.random.beta((dim - 1) / 2, (dim - 1) / 2)
            w[i] = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.uniform()
            if kappa * w[i] + (dim - 1) * np.log(1 - b * w[i]) - c >= np.log(u):
                break

    v = np.random.randn(num_samples, dim - 1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    result[:, :-1] = np.sqrt(1 - w**2).reshape(-1, 1) * v
    result[:, -1] = w

    # Rotate result to have the mean direction mu
    if not np.allclose(mu, np.array([0]*(dim-1) + [1])):
        mu = mu / norm(mu)
        z_axis = np.zeros(dim)
        z_axis[-1] = 1
        u = mu - z_axis
        u /= norm(u)
        
        # Householder reflection matrix
        H = np.eye(dim) - 2 * np.outer(u, u)
        result = result @ H.T
    
    return result



###
mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
kappa = 10
num_samples = 1000
X = sample_vmf(mu, kappa, num_samples)
###


intdim_k_repeated = repeated(intrinsic_dim_scale_interval, X, mode='bootstrap', nb_iter=100, # nb_iter for bootstrapping, 
                            verbose=1, k1=10, k2=20)

intdim_k_repeated = np.array(intdim_k_repeated)
plt.hist(intdim_k_repeated.mean(axis=1))
plt.xlabel('Mean Intrinsic Dimension')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Intrinsic Dimensions')
plt.show()
