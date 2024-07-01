import numpy as np
import matplotlib as plt

import math

def calc_metric(n, k, result_full, result):
    indices = np.arange(1, n + 1)

    x = 0
    y = 0
    q = n // k  # Using integer division

    for index in indices:
        # Adjust index for zero-based indexing
        zbd = index - 1
        
        if zbd < n / k and result_full[zbd] > result[k]:
            x += 1
            print("Condition 1; index = " + str(index) + "; value = " + str(result_full[zbd]))
        elif zbd >= (n * (k - 1)) / k and result_full[zbd] < result[n-q-1]:
            x += 1
            print("Condition 2; index = " + str(index) + "; value = " + str(result_full[zbd]))
        elif n / k <= zbd < (n * (k - 1)) / k:
            # Calculate the indices for comparison
            lower_index = q * (math.ceil(k * index / n) - 2)
            upper_index = q * math.ceil(k * index / n)
            
            # Ensure indices are within bounds and convert to integers
            lower_index = int(max(0, lower_index))
            upper_index = int(min(n - 1, upper_index))
            
            if result[lower_index] > result_full[zbd] > result[upper_index]:
                x += 1
                print("Condition 3; index = " + str(index) + "; value = " + str(result_full[zbd]))

    prop = x / n
    print("Number correct: " + str(x))
    print("Proportion: " + str(prop))

    # Display results
    plt.semilogy(result_full, '.', label='Full Matrix')
    plt.semilogy(result, 'x', label='Sampled Matrix')
    plt.legend()
    plt.show()
    return

