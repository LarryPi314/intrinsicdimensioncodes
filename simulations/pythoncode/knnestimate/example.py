import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_spherical_data(n_samples, n_dimensions, radius=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random points on a unit sphere
    X = np.random.randn(n_samples, n_dimensions)
    X /= np.linalg.norm(X, axis=1)[:, np.newaxis]  # Normalize each row to have unit norm
    X *= radius  # Scale to desired radius
    
    return X

# Parameters for data generation
n_samples = 729
n_dimensions = 4  # Number of dimensions (typically 2 or 3 for visualization)
radius = 1.0  # Radius of the spherical distribution

# Generate synthetic data on a sphere
X = generate_spherical_data(n_samples, n_dimensions, radius=radius)

# Number of neighbors for kNN
k = 9

# Compute distances to the k-nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)

# Calculate the intrinsic dimensionality estimate with epsilon
epsilon = 1e-10  # Small constant to avoid division by zero
log_ratios = np.log(distances[:, k-1][:, np.newaxis] / (distances[:, :k-1] + epsilon))
mean_log_ratios = np.mean(log_ratios)
intrinsic_dimension_estimate = (mean_log_ratios / (k - 1)) ** -1

print(f'Estimated intrinsic dimensionality: {intrinsic_dimension_estimate:.2f}')

