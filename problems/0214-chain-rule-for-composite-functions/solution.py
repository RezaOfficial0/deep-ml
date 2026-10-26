import numpy as np

_FUNCS = {
    'square': (lambda u: u ** 2,     lambda u: 2 * u),
    'sin':    (lambda u: np.sin(u),  lambda u: np.cos(u)),
    'exp':    (lambda u: np.exp(u),  lambda u: np.exp(u)),
    'log':    (lambda u: np.log(u),  lambda u: 1 / u),
}

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    u = x
    grad = 1.0  # d(identity)/dx; an empty list returns 1.0
    for name in reversed(functions):  # innermost function first
        if name not in _FUNCS:
            raise ValueError(f"Unknown function: {name}")
        f, df = _FUNCS[name]
        if name == 'log' and u <= 0:
            raise ValueError(f"log undefined at {u}")
        grad *= df(u)  # local derivative at this function's input
        u = f(u)       # pass the output on to the next function
    return float(grad)