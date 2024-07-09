
import numpy as np

def sample_uniform_sphere(num_samples):
    """
    Generate samples from a uniform sphere in 3D.

    :param num_samples: number of samples to generate
    :return: array of shape (num_samples, 3) containing the generated samples
    """
    samples = np.random.randn(num_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    return samples
