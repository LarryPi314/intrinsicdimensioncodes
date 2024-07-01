from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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
def intrinsic_dim_scale_interval(X, k1=10, k2=20):
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




def generate_spherical_data(n_samples, n_dimensions, radius=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random points on a unit sphere
    X = np.random.randn(n_samples, n_dimensions)
    X /= np.linalg.norm(X, axis=1)[:, np.newaxis]  # Normalize each row to have unit norm
    X *= radius  # Scale to desired radius
    
    return X

# Generate uniformly distributed spherical data in n_dimensions
np.random.seed(0)
n_samples = 729
n_dimensions = 4  # Specify the number of dimensions
radius = 1.0  # Radius of the spherical distribution

# Generate synthetic data on a sphere
X = generate_spherical_data(n_samples, n_dimensions, radius=radius, random_state=None)

# Perform repeated intrinsic dimension estimation
intdim_k_repeated = repeated(intrinsic_dim_scale_interval, X, mode='bootstrap', nb_iter=100, verbose=1, k1=10, k2=20)

# Convert results to numpy array
intdim_k_repeated = np.array(intdim_k_repeated)

# Plot histogram of mean intrinsic dimensions
plt.hist(intdim_k_repeated.mean(axis=1), bins=30, edgecolor='black')
plt.xlabel('Mean Intrinsic Dimension')
plt.ylabel('Frequency')
plt.title(f'Histogram of Mean Intrinsic Dimensions (Uniformly Distributed Spherical Data in {n_dimensions}D)')
plt.show()
