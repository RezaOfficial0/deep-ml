def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float] | int:
    # Bos matris veya herhangi bir satirin uzunlugu b ile uyusmuyorsa -1
    if not a or any(len(row) != len(b) for row in a):
        return -1

    return [sum(x * y for x, y in zip(row, b)) for row in a]