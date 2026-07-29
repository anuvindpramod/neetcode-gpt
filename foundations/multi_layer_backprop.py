import numpy as np
from typing import List


class Solution:
    
    def ReLu(self,x):
        return np.maximum(0,x)

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x=np.array(x)
        W1=np.array(W1)
        b1=np.array(b1)
        W2=np.array(W2)
        b2=np.array(b2)
        y_true=np.array(y_true)

        z1=np.matmul(x,W1.T)+b1 #(1,2)
        y1=self.ReLu(z1)#(1,2)
        y2=np.matmul(y1,W2.T)+b2 #(1)

        # Loss: MSE = mean((predictions - y_true)^2)
        loss=np.mean((y2-y_true)**2)
        n=len(y_true) if y_true.ndim > 0 else 1
        dy2=2*(y2-y_true)/n #(1)
        db2=dy2
    
        dW2=dy2.reshape(-1,1) @ y1.reshape(1,-1)
        

        dy1 = dy2 @ W2

        dz1 = dy1 * (z1 > 0)

        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)

        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }



        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
