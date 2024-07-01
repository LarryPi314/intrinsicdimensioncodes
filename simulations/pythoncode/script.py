import math
import numpy as np
import matplotlib.pyplot as plt
from average_full_mat import average_full_mat
from average_repmat import average_repmat
from create_new_scheme_points import create_new_scheme_points
from generate_Cauchy_kernel import generate_cauchy_kernel
from generate_Gaussian_kernel import generate_gaussian_kernel

# Assuming the previous functions are already defined: 
# generate_gaussian_kernel, generate_cauchy_kernel, create_new_scheme_points, average_full_mat, average_repmat

n=125
k=5
d=4
l=1/10


print("<==== RUNNING CODE ====>")
# Average the eigenvalues of the "full matrix"
result_full = average_full_mat(10, 'sphere', 'Gaussian', n, d, l) 

# Average the eigenvalues of the "sampled matrix"
result = average_repmat(128000, 'sphere', 'Gaussian', n, k, d, l) 

# Display results
plt.semilogy(result_full, '.', label='Full Matrix')
plt.semilogy(result, 'x', label='Sampled Matrix')
plt.legend()
plt.show()


# Assume result_full, result, n, and k are defined
indices = np.arange(1, n + 1)

x = 0
y = 0

for index in indices:
    # Adjust index for zero-based indexing
    zero_based_index = index - 1
    
    if index < n / k and result_full[zero_based_index] > result[1]:
        x += 1
    elif index > n * (k - 1) / k and result_full[zero_based_index] < result[k - 2]:
        x += 1
    elif n / k < index < n * (k - 1) / k and result[math.ceil(k * index / n) - 2] < result_full[zero_based_index] < result[math.ceil(k * index / n) - 1]:
        x += 1

prop = x / n
print("Proportion: " + str(prop))
