import pytest
import numpy as np
import matplotlib.pyplot as plt

from disp_disk_compression import run

def test_linear_convergence():
    result = []
    diff = []
    element_counts = np.array([4, 8, 16, 32, 64, 128])#, 256])
    result.append(run(1.0, 0.25, 1.0, 0.2, n_elements = element_counts[0]))
    for n_elements in element_counts[1:]:
        result.append(run(1.0, 0.25, 1.0, 0.2, n_elements, 1))
        difference = np.sum(np.abs((result[-1] - result[-2]) / result[-1]))
        diff.append(difference)
    plt.plot(np.log(1.0 / element_counts[1:]), np.log(diff), '*-')
    plt.show()
