import numpy as np
import matplotlib.pyplot as plt
from average_full_mat import average_full_mat
from average_repmat import average_repmat
from create_new_scheme_points import create_new_scheme_points
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

# Assuming the previous functions are already defined: 
# generate_gaussian_kernel, generate_cauchy_kernel, create_new_scheme_points, average_full_mat, average_repmat

# Average the eigenvalues of the "full matrix"
result_full = average_full_mat(8000, 'sphere', 'Gaussian', 729, 2) 

# Average the eigenvalues of the "sampled matrix"
result = average_repmat(8000, 'sphere', 'Gaussian', 729, 9, 2) #8000, 49, 7

# Display results
plt.semilogy(result_full, '.', label='Full Matrix')
plt.semilogy(result, 'x', label='Sampled Matrix')
plt.legend()
plt.show()
