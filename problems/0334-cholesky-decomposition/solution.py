import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.

    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)

    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    A = np.array(A, dtype=float)

    # Kare ve simetrik olmali
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.allclose(A, A.T):
        return -1

    n = A.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        # Kosegen: karekok icindeki deger pozitif degilse matris PD degil
        d = A[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0:
            return -1
        L[j, j] = np.sqrt(d)

        # Kosegen alti
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]

    return L.tolist()