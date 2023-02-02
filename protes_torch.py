import numpy as np
import teneva
from time import perf_counter as tpc
import torch


from utils import ind_tens_max_ones


def protes_torch(f, n, M, K=50, k=5, k_gd=100, r=5, lr=1.E-2, is_max=False, constr=False, log=False):
    """Tensor optimization based on sampling from the probability TT-tensor.

    Method PROTES (PRobability Optimizer with TEnsor Sampling) for optimization
    of the multidimensional arrays and  discretized multivariable functions
    based on the tensor train (TT) format.

    Args:
        f (function): the target function "f(I)", where input "I" is a 2D
            np.ndarray of the shape "[samples, d]" ("d" is a number of
            dimensions of the function's input). The function should return 1D
            np.ndarray of the length equals to "samples" (the values of the
            target function for all provided multi-indices).
        n (list of int): tensor size for each dimension.
        M (int): the number of allowed requests to the objective function.
        K (int): the batch size for optimization.
        k (int): number of selected candidates for all batches (< K).
        k_gd (int): number of GD iterations for each batch.
        r (int): TT-rank of the constructed probability TT-tensor.
        lr (float): learning rate for GD.
        is_max (bool): if is True, then maximization will be performed.
        constr (bool): if flag is set, then the constraint will be used (it
            works now only for special optimal control problem TODO).
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed every step.

    Returns:
        tuple: multi-index "n_opt" (list of the length "d") corresponding to
        the found optimum of the tensor and the related value "y_opt" (float).

    """

    time, i_opt, y_opt, m = tpc(), None, None, 0
    P = _generate_initial(n, r)

    while True:
        I_full = _sample(P, min(K, M-m))
        y_full = f(I_full)
        m += y_full.shape[0]

        ind_opt = np.argmax(y_full) if is_max else np.argmin(y_full)
        i_opt_curr, y_opt_curr = I_full[ind_opt, :], y_full[ind_opt]

        is_new = y_opt is None
        is_new = is_new or is_max and y_opt < y_opt_curr
        is_new = is_new or not is_max and y_opt > y_opt_curr
        if is_new:
            i_opt, y_opt = i_opt_curr, y_opt_curr

            if log:
                text = ''
                text += f'Evals : {m:-7.1e} | '
                text += f'Opt : {y_opt:-14.7e} | '
                text += f'Time : {tpc()-time:-14.3f}'
                print(text)

        if m >= M:
            break

        ind = np.argsort(y_full)
        ind = (ind[::-1] if is_max else ind)[:k]
        _optimize(P, I_full[ind, :], y_full[ind], k_gd, lr)

    return i_opt, y_opt


def _generate_initial(n, r, constr=False):
    """Build initial TT-tensor for probability."""
    if constr:
        Y = ind_tens_max_ones(d, 3, r)

    else:
        d = len(n)
        rs = [1] + [r]*(d-1) + [1]
        Y = []
        for i in range(d):
            Y.append(np.random.random(size=(rs[i], n[i], rs[i+1])))

    return [torch.tensor(G, requires_grad=True) for G in Y]


def _get_many(Y, I):
    """Compute the elements of the TT-tensor (in pytorch) on many indices."""
    I = np.asanyarray(I, dtype=int)
    Q = Y[0][0, I[:, 0], :]
    for i in range(1, len(Y)):
        Q = torch.einsum('kq,qkp->kp', Q, Y[i][:, I[:, i], :])
    return Q[:, 0]


def _optimize(P, I_trn, y_trn, k_gd, lr, optimizer=torch.optim.Adam):
    """Perform several GD steps for TT-tensor."""
    def loss_func(P_curr):
        p = _get_many(P_curr, I_trn)
        l = -torch.log(p)
        l = torch.sum(l)
        return l

    opt = optimizer([G for G in P if G.requires_grad], lr)
    for _ in range(int(k_gd)):
        opt.zero_grad()
        loss = loss_func(P)
        loss.backward(retain_graph=True)
        opt.step()


def _sample(Y, K):
    """Generate K samples according to given probability TT-tensor."""
    d = len(Y)
    n = [G.shape[1] for G in Y]
    I = torch.zeros((K, d), dtype=torch.int)
    Z = [None] * (d+1)

    Z[-1] = torch.ones(1, dtype=torch.double)
    for j in range(d-1, 0, -1):
        Z[j] = torch.sum(Y[j], dim=1) @ Z[j+1]
        Z[j] = Z[j].reshape(-1)

    p = Y[0] @ Z[1]
    p = p.flatten()
    p[p < 0] = 0.
    p = p / p.sum()

    ind = torch.multinomial(p, K, replacement=True)

    Z[0] = Y[0][0, ind, :]
    I[:, 0] = ind

    for j, G in enumerate(Y[1:], start=1):
        p = torch.einsum('ma,aib,b->mi', Z[j-1], G, Z[j+1])
        p[p < 0] = 0.

        ind = torch.tensor([
            torch.multinomial(pi/pi.sum(), 1, replacement=True) for pi in p])

        Z[j] = torch.einsum("il,lij->ij", Z[j-1], G[:, ind])
        I[:, j] = ind

    return I
