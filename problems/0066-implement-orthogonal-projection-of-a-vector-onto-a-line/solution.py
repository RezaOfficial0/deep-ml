
def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.
    """
    dot_product = sum(v[i] * L[i] for i in range(len(v)))
    magnitude_squared = sum(x * x for x in L)

    if magnitude_squared == 0:
        raise ValueError("L must be a non-zero vector")

    projection = [
        (dot_product / magnitude_squared) * x
        for x in L
    ]

    return projection
