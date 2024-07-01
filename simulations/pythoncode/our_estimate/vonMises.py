
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
from scipy.special import iv
from numpy.linalg import norm

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

def plot_vmf_samples(ax, x, y, z, mu, kappa, num_samples=100):
    samples = sample_vmf(mu, kappa, num_samples)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, alpha=0.2)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', s=5)
    ax.scatter(mu[0], mu[1], mu[2], c='r', s=30)  # Red dot to indicate the mean direction
    ax.set_aspect('equal')
    ax.view_init(azim=-130, elev=0)
    ax.axis('off')
    ax.set_title(rf"$\kappa={kappa}$")

#n_grid = 100
#u = np.linspace(0, np.pi, n_grid)
#v = np.linspace(0, 2 * np.pi, n_grid)
#x = np.outer(np.cos(v), np.sin(u))
#y = np.outer(np.sin(v), np.sin(u))
#z = np.outer(np.ones_like(u), np.cos(u))



#mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
#fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4), subplot_kw={"projection": "3d"})
#plot_vmf_samples(axes[0], x, y, z, mu, 5)
#plot_vmf_samples(axes[1], x, y, z, mu, 20)
#plot_vmf_samples(axes[2], x, y, z, mu, 100)
#plt.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, wspace=0.)
#plt.show()


