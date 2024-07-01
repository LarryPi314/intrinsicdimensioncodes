import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
n_dimensions = 4  # Number of dimensions for visualization (2 or 3 recommended)
radius = 1.0  # Radius of the spherical distribution
random_state = 0  # Random state for reproducibility

# Generate synthetic data on a sphere
X = generate_spherical_data(n_samples, n_dimensions, radius=radius, random_state=random_state)

# Plotting the spherical data in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='o', alpha=0.6)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Spherical Data Scatter Plot')

plt.show() 
