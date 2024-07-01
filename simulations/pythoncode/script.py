import math
import numpy as np
import matplotlib.pyplot as plt
from average_full_mat import average_full_mat
from average_repmat import average_repmat
from create_new_scheme_points import create_new_scheme_points
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

# Average the eigenvalues of the "full matrix"
result_full = average_full_mat(10, 'sphere', 'Gaussian', 729, 2) 

# Average the eigenvalues of the "sampled matrix"
result = average_repmat(50, 'sphere', 'Gaussian', 729, 9, 2) # adjust n, k, d hyperparameters as needed. 

# Display results
plt.semilogy(result_full, '.', label='Full Matrix')
plt.semilogy(result, 'x', label='Sampled Matrix')
plt.legend()
plt.show()


