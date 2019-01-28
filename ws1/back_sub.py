import numpy as np
from numpy.random import randint, random
np.random.seed(42)


def back_sub_row_wise(U, b):
    """
    Algorithm from Bjork 1.2.1
    :param U: An upper triangular matrix (np, scipy array)
    :param b: right-side constant array
    :return: solution vector x
    """
    assert U.shape[0] == U.shape[1], "U needs to be a square matrix"
    assert np.sum([np.abs(U[i, j]) for i in range(len(U)) for j in range(i)]) == 0, "U needs to be a upper triangular"
    assert np.prod(np.diagonal(U)) != 0, "U needs to be invertible"
    n = U.shape[0]
    x = np.zeros((n, ))
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:]))/U[i, i]
    return x


def back_sub_col_wise(U, b):
    """
    Algorithm from Bjork 1.2.2
    :param U: An upper triangular matrix (np, scipy array)
    :param b: right-side constant array
    :return: solution vector x
    """
    assert U.shape[0] == U.shape[1], "U needs to be a square matrix"
    assert np.sum([np.abs(U[i, j]) for i in range(len(U)) for j in range(i)]) == 0, "U needs to be a upper triangular"
    assert np.prod(np.diagonal(U)) != 0, "U needs to be invertible"
    n = U.shape[0]
    x = np.copy(b)
    for i in range(n-1, -1, -1):
        x[i] = x[i]/U[i, i]
        x[:i] = x[:i] - x[i]*U[:i, i]
    return x

m = 10
U = np.triu(randint(100, size=(m, m)))*1.0
b = randint(10, size=(m,))*1.0
res = back_sub_row_wise(U, b)
res2 = back_sub_col_wise(U, b)
print(res)
print(res2)
print(b)
print(np.dot(U, res))