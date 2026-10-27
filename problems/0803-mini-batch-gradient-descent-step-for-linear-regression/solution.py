import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """

    X_b = X[batch_indices]
    y_b = y[batch_indices]
    b = len(batch_indices)

    guess = X_b @ weights + bias
    error = guess - y_b

    grad_w = 2 * X_b.T @ error / b
    grad_b = 2 * np.sum(error) / b

    new_weights = weights - lr * grad_w
    new_bias = bias - lr * grad_b

    return np.append(new_weights, new_bias)
