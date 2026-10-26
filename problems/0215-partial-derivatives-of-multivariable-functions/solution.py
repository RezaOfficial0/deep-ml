import numpy as np

_GRADIENTS = {
    # f = x²y + xy²
    'poly2d':        (2, lambda x, y: (2*x*y + y**2, x**2 + 2*x*y)),
    # f = e^(x+y)
    'exp_sum':       (2, lambda x, y: (np.exp(x + y), np.exp(x + y))),
    # f = x·sin(y)
    'product_sin':   (2, lambda x, y: (np.sin(y), x * np.cos(y))),
    # f = x²y + yz²
    'poly3d':        (3, lambda x, y, z: (2*x*y, x**2 + z**2, 2*y*z)),
    # f = (x - y)²
    'squared_error': (2, lambda x, y: (2*(x - y), -2*(x - y))),
}

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    if func_name not in _GRADIENTS:
        raise ValueError(f"Unknown function: {func_name}")
    arity, grad = _GRADIENTS[func_name]
    if len(point) != arity:
        raise ValueError(f"{func_name} expects {arity} coordinates, got {len(point)}")
    return tuple(float(g) for g in grad(*point))