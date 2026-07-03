import numpy as np

from xcps_ai.reduce import default_max_lag, diagonal_average


def test_diagonal_average_uses_upper_diagonals():
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    frame = diagonal_average(matrix)

    assert frame["lag"].tolist() == [0, 1, 2]
    assert frame["n_pairs"].tolist() == [3, 2, 1]
    assert frame["g2"].tolist() == [5.0, 4.0, 3.0]


def test_default_max_lag_applies_cutoff():
    assert default_max_lag(100, 10) == 89
    assert default_max_lag(100, 0) == 99
