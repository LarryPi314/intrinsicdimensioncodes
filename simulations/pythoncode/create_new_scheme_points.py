import numpy as np

def create_new_scheme_points(X, n, k, d):
    """
    Generate new scheme points based on the given parameters.
    This function creates the reduced dimension points (vectors) as outlined
    in Dr. Lepilov's paper. The reduced dimension points are generated based on
    a probability mass function with pre-computed parameters which transform randomly
    sampled points from the input data array.

    Parameters:
    X (numpy.ndarray): Input data array.
    n (int): Number of points to select from.
    k (int): Number of points to select for transformation.
    d (float): Transformation factor.

    Returns:
    numpy.ndarray: Transformed data array.
    """

    if (n == 49 and k == 7):
        p = np.array([0.411659, 0.568099, 0.0202407, 0.0000014709])
        v = np.array([4.8651, 9.68274, 24.5194, 130.899])

        c = np.cumsum(np.concatenate(([0], p)))
        c = c / c[-1]  # Normalize cumulative probabilities
        i = np.searchsorted(c, np.random.rand(k), side='right') - 1 # Find the indices of the random numbers in the cumulative probabilities

        coeffs = v[i] # Select the vectors based on the indices
        coeffs[1:] = coeffs[0]

        dupled_coeffs = np.tile(coeffs, (X.shape[1], 1))
        factor = coeffs[0]

        # Apply transformation
        Xk = (1. / dupled_coeffs) ** (1 / d) * X[np.random.randint(0, 49, 7), :].T

    elif (n == 729 and k == 9):
        p = np.array([0.364337, 0.579978, 0.0555154, 0.000169223, 0.00000000032717])
        v = np.array([57.2048, 101.685, 507.901, 199.89, 2721.69])

        c = np.cumsum(np.concatenate(([0], p)))
        c = c / c[-1]  # Normalize cumulative probabilities
        i = np.searchsorted(c, np.random.rand(k), side='right') - 1

        coeffs = v[i]
        coeffs[1:] = coeffs[0]

        dupled_coeffs = np.tile(coeffs, (X.shape[1], 1))

        # Apply transformation
        Xk = (1. / dupled_coeffs) ** (1 / d) * X[np.random.randint(0, 729, 9), :].T
    
    elif(n == 125 and k == 5): #added new n, k cases
        p = np.array([0.00353251, 0.977753, 0.0187149])
        v = np.array([-104.149, 29.3087, 144.874])

        c = np.cumsum(np.concatenate(([0], p)))
        c = c / c[-1]  # Normalize cumulative probabilities
        i = np.searchsorted(c, np.random.rand(k), side='right') - 1

        coeffs = v[i]
        coeffs[1:] = coeffs[0]

        dupled_coeffs = np.tile(coeffs, (X.shape[1], 1))

        # Apply transformation
        Xk = (1. / dupled_coeffs) ** (1 / d) * X[np.random.randint(0, 125, 5), :].T
    return Xk.T

# intrinsic dimension d = 2 for sphere and manifold.