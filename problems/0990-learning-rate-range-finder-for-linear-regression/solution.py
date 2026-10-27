import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    w = np.array(w0, dtype=float).copy()
    m = len(X)

    # n_steps learning rates, spread evenly on a log scale from 10**a to 10**b
    learning_rates = np.logspace(a, b, n_steps)

    losses = []
    for lr in learning_rates:
        # one full-batch step (use all rows)
        error = X @ w - y
        gradient = 2 * X.T @ error / m
        w = w - lr * gradient

        # measure the loss AFTER the step
        loss = np.mean((X @ w - y) ** 2)
        losses.append(float(loss))

    return losses
