import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """


    n_samples = len(X)
    w = weights.astype(float).copy()
    for i in range(n_iter):
        idx = i % n_samples

        X_i = X[idx]
        y_i = y[idx]

        guess = X_i @ w
        error = guess - y_i
        gradient = 2 * error * X_i

        w = w - learning_rate * gradient

    return w.tolist()
