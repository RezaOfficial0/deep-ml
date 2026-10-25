import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.

    Args:
        A: An m x n matrix represented as list of lists

    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # Modified Gram-Schmidt: projeksiyonu guncel v uzerinden cikar
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        if np.isclose(R[j, j], 0.0):
            raise ValueError("Sutunlar lineer bagimli: Gram-Schmidt QR tanimsiz")

        Q[:, j] = v / R[j, j]

    return Q.tolist(), R.tolist()