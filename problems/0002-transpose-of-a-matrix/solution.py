def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    if not a:
        return []

    return [list(col) for col in zip(*a)]