import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    x = x.astype(float)
    numerical_grad = np.zeros_like(x)

    # nudge each number in x a tiny bit up and down, one at a time
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus.flat[i] += epsilon
        x_minus.flat[i] -= epsilon

        # slope = (height a bit to the right - height a bit to the left) / distance
        numerical_grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    # how different are the two gradients, as a fraction of their size
    diff = np.linalg.norm(numerical_grad - analytical_grad)
    total = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = 0.0 if total == 0 else diff / total

    return numerical_grad, relative_error