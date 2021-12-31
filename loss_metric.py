import numpy as np
from sklearn.preprocessing import StandardScaler


def loss_function(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    n = x.shape[0]

    # Appending a column of 1's to x (to include the bias term)
    x = np.append(x, np.ones((n, 1)), axis=1)

    # Reshaping y to match x's dimensionality
    y = y.reshape(n, 1)  # y shape - (N x 1)

    # return np.sum(np.square(x@w - y), axis=0)
    # return np.sum(np.square(np.power(np.e, x @ w) - y), axis=0)
    return np.mean(np.square((x @ w) - y), axis=0)
