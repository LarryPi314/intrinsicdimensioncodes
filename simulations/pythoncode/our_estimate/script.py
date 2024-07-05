import numpy as np
import matplotlib.pyplot as plt
from average_full_mat import average_full_mat
from average_repmat import average_repmat
from create_new_scheme_points import create_new_scheme_points
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel
from comparison_metric import calc_metric

n=625
k=5
d=4
l=.0075


print("<==== RUNNING CODE ====>")
# Average the eigenvalues of the "full matrix"
result_full = average_full_mat(10, 'vonMises', 'Gaussian', n, d, l) 

# Average the eigenvalues of the "sampled matrix"
result = average_repmat(50000, 'vonMises', 'Gaussian', n, k, d, l) 

# Find proportion of eigenvalues which fall within quantile bounds
<<<<<<< HEAD
calc_metric(n, k, result_full, result)
=======


# Display results
plt.semilogy(result_full, '.', label='Full Matrix')
plt.semilogy(result, 'x', label='Sampled Matrix')
plt.legend()
plt.show()
>>>>>>> origin/aaron
