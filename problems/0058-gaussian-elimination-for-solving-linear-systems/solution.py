import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Forward elimination
    for k in range(n - 1):
        # Partial pivoting: k. sutunda (k. satir ve alti) mutlak degeri en buyuk eleman
        p = k + np.argmax(np.abs(A[k:, k]))
        if np.isclose(A[p, k], 0.0):
            raise ValueError("Matris tekil (singular)")

        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]

        # Pivotun altindaki tum satirlari tek seferde sifirla
        factors = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(factors, A[k, k:])
        b[k + 1:] -= factors * b[k]

    if np.isclose(A[n - 1, n - 1], 0.0):
        raise ValueError("Matris tekil (singular)")

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x
