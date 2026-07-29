import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import torch
import torch.nn as nn


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        w=np.zeros(X.shape[1])
        b=0.0
        n=X.shape[0]
        for _ in range(epochs):
            y_hat=X @ w + b
            loss=y_hat-y

            dw=2/n * (X.T @ loss)
            db=2/n * (np.sum(loss))

            w-=lr*dw
            b-=lr*db

        return (np.round(w,5),round(b,5))
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        pass
