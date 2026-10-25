import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)

    r = b - A @ x          # residual
    p = r.copy()           # ilk arama yonu = residual
    rs_old = r @ r

    for _ in range(n):
        if np.sqrt(rs_old) < tol:
            break

        Ap = A @ p
        alpha = rs_old / (p @ Ap)      # p yonunde optimal adim
        x += alpha * p
        r -= alpha * Ap                # residual'i A@x yeniden hesaplamadan guncelle

        rs_new = r @ r
        beta = rs_new / rs_old
        p = r + beta * p               # yeni yon, oncekilere A-ortogonal
        rs_old = rs_new

    return x