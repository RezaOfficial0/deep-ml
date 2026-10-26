import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    g = np.asarray(gradient, dtype=float)
    magnitude = float(np.linalg.norm(g))

    if magnitude == 0.0:
        # A stationary point has no defined direction, so return zero vectors
        direction = np.zeros_like(g)
    else:
        direction = g / magnitude

    return {
        "magnitude": magnitude,
        "direction": direction.tolist(),
        "descent_direction": (-direction).tolist(),
    }