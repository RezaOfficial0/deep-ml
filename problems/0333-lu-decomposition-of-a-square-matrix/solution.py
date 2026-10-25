import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.

    Args:
        A: Square matrix as a list of lists

    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("Kare matris olmali")

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # U'nun i. satiri
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]

        if np.isclose(U[i, i], 0.0):
            raise ValueError("Sifir pivot: pivotsuz Doolittle LU bu matris icin yok")

        # L'nin i. sutunu (kosegen altı)
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]

    return L, U