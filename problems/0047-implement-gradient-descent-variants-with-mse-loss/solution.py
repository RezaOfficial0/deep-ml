import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m = len(X)

    # how many rows we use before each step
    if method == 'batch':
        group_size = m
    elif method == 'stochastic':
        group_size = 1
    elif method == 'mini_batch':
        group_size = batch_size
    else:
        raise ValueError("method must be 'batch', 'stochastic', or 'mini_batch'")

    w = weights.astype(float).copy()

    for epoch in range(n_epochs):
        # no shuffling: go through rows in order
        for start in range(0, m, group_size):
            X_group = X[start : start + group_size]
            y_group = y[start : start + group_size]

            guess = X_group @ w
            error = guess - y_group
            gradient = 2 * X_group.T @ error / group_size   # the 2 is the fix
            w = w - learning_rate * gradient

    return w
