import numpy as np


from .bm_opt_tens import BmOptTens


class BmOptTensMmul(BmOptTens):
    def __init__(self, size=2, rank=7, only2=False, name='mmul', E=None):
        if E is None:
            E = [-1, 0, 1]    # Possible items of the factor matrices
        T = tensor_generate(size, size, size)
        d = (2 if only2 else 3) * T.shape[0] * rank

        super().__init__(d, len(E), name+f'-{size}')

        self.desc = """
            Problem for fast 2x2 matrix multiplications.
        """

        if size == 2 and only2 == False:
            self.i_min = [
                2, 1, 2, 1, 2, 0, 1,
                1, 1, 1, 1, 2, 1, 2,
                1, 2, 1, 1, 1, 2, 1,
                2, 2, 1, 2, 1, 1, 0,

                2, 2, 1, 0, 1, 2, 1,
                1, 1, 2, 1, 1, 2, 1,
                1, 1, 1, 2, 1, 1, 2,
                2, 1, 0, 1, 2, 1, 2,

                2, 1, 1, 2, 0, 1, 2,
                1, 1, 2, 1, 2, 1, 1,
                1, 2, 1, 2, 1, 1, 1,
                2, 0, 2, 1, 1, 2, 1,
            ]

        self.y_min = 0.

        self.E = E
        self.size = size
        self.rank = rank
        self.only2 = only2
        self._func = loss_build(T, E, rank, only2)

    def recover(self, i):
        x = ind_to_poi(i, self.E)
        print(x)

        if self.only2:
            U, V = factor_from_poi(x, self.rank, True)
            W = factor_recover(U, V, T)
        else:
            U, V, W = factor_from_poi(x, self.rank, False)

        return U, V, W

    def _f(self, i):
        return self._func(i)


def factor_from_poi(x, q, only2=False, order_spec=False):
    """Build canonical rank-q factors from flatten "x"."""
    k = 2 if only2 else 3
    n = x.size // (k * q)

    if order_spec and not only2:
        raise ValueError('Is not supported')
    elif order_spec:
        U = np.array([x[n*2*j:n*2*j+n] for j in range(q)]).T
        V = np.array([x[n*2*j+n:n*2*j+2*n] for j in range(q)]).T
    else:
        U = x[:n*q].reshape((n, q))
        V = x[n*q:2*n*q].reshape((n, q))

    if only2:
        return U, V

    W = x[2*n*q:].reshape((n, q))

    return U, V, W


def factor_recover(U, V, T):
    """Build 3th factor matrix from 2 given factor matrices and 3D tensor."""
    n = T.shape[-1]
    q = U.shape[-1]

    A = np.einsum('nr,mr->nmr', U, V).reshape(-1, q)
    R = T.reshape(-1, n)

    W = np.linalg.lstsq(A, R, rcond=-1)[0].T

    return W


def ind_to_poi(I, E=[-1, 0, 1]):
    """Transform tensor multi-index into point from discrete values "E"."""
    return np.asarray(E)[list(I)]


def tensor_generate(a, b, c):
    """Generate the matrix multiplication tensor T_{a, b, c}."""
    T = np.full((a*b, b*c, c*a), 0, dtype=int)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i * b + j][j * c + k][k + i * c] = 1
    return T


def loss_build(T_real, E, q, only2=True, order_spec=False, fast=False):
    if fast:
        T = T_real.reshape(-1, T_real.shape[-1])

    def loss(i):
        x = ind_to_poi(i, E)
        if only2:
            U, V = factor_from_poi(x, q, True, order_spec)
            if not fast:
                W = factor_recover(U, V, T_real)
        else:
            U, V, W = factor_from_poi(x, q, False, order_spec)

        if only2 and fast:
            # NOTE: this code may be invalid now (compare with fast=False)
            A = np.einsum('nr,mr->nmr', U, V).reshape(-1, U.shape[-1])
            Q = np.linalg.qr(A)[0]
            D = T - Q @ (Q.T @ T)
            e = np.linalg.norm(D.reshape(-1))
        else:
            T_appr = np.einsum('nr,mr,sr->nms', U, V, W)
            e = np.linalg.norm(T_appr.reshape(-1) - T_real.reshape(-1))

        return e

    return loss
