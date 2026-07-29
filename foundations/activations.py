import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        sig_out=1/(1+(np.exp(-z)))
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        return np.round(sig_out,5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        relu_out=np.maximum(0,z)
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return relu_out
