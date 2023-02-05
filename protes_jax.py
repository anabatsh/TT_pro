import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from time import perf_counter as tpc


from utils import ind_tens_max_ones


def protes_jax(f, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, info={}, i_ref=None, is_max=False, constr=False, log=False, log_ind=False):
    """Tensor optimization based on sampling from the probability TT-tensor.

    Method PROTES (PRobability Optimizer with TEnsor Sampling) for optimization
    of the multidimensional arrays and  discretized multivariable functions
    based on the tensor train (TT) format. This is the "jax" version.

    Args:
        f (function): the target function "f(I)", where input "I" is a 2D jax
            array of the shape "[samples, d]" ("d" is a number of dimensions
            of the function's input). The function should return 1D jax array
            on the CPU or GPU of the length equals to "samples" (the values of
            the target function for all provided multi-indices).
        n (list of int): tensor size for each dimension.
        m (int): the number of allowed requests to the objective function.
        k (int): the batch size for optimization.
        k_top (int): number of selected candidates for all batches (< k).
        k_gd (int): number of GD iterations for each batch.
        lr (float): learning rate for GD.
        r (int): TT-rank of the constructed probability TT-tensor.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
        i_ref (list of int): optional multi-index, in which the values of the
            probabilistic tensor will be stored during iterations (the result
            will be available in the info dictionary in the 'y_ref_list' field).
        is_max (bool): if is True, then maximization will be performed.
        constr (bool): if flag is set, then the constraint will be used (it
            works now only for special optimal control problem TODO).
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed every step.
        log_ind (bool): if flag is set and "log" is True, then the current
            optimal multi-index will be printed every step.

    Returns:
        tuple: multi-index "i_opt" (list of the length "d") corresponding to
        the found optimum of the tensor and the related value "y_opt" (float).

    """
    time = tpc()
    info.update({'m': 0, 't': 0, 'i_opt': None, 'y_opt': None, 'is_max': is_max,
        'm_opt_list': [], 'y_opt_list': [], 'm_ref_list': [], 'p_ref_list': [],
        'p_opt_ref_list': [], 'p_top_ref_list': []})

    optim = optax.adam(lr)
    sample = jax.jit(jax.vmap(_sample_one, (None, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood_one, (None, 0)))

    P = _generate_initial(n, r, constr)
    opt_state = optim.init(P)
    rng = jax.random.PRNGKey(42)

    @jax.jit
    def loss(P_cur, I_cur):
        return jnp.mean(-likelihood(P_cur, I_cur))

    @jax.jit
    def optimize(P, I_cur, opt_state):
        grads = jax.grad(loss)(P, I_cur)
        updates, opt_state = optim.update(grads, opt_state)
        P = eqx.apply_updates(P, updates)
        return P, opt_state

    while True:
        rng, key = jax.random.split(rng)
        key_curr = jax.random.split(key, k)
        I = sample(P, key_curr)

        y = f(I)
        info['m'] += y.shape[0]

        is_new = _check(I, y, info)

        if info['m'] >= m:
            break

        ind = jnp.argsort(y, kind='stable')
        ind = (ind[::-1] if is_max else ind)[:k_top]

        for _ in range(k_gd):
            P, opt_state = optimize(P, I[ind, :], opt_state)

        info['m_ref_list'].append(info['m'])
        info['p_opt_ref_list'].append(_get_one(P, info['i_opt']))
        info['p_top_ref_list'].append(_get_one(P, I[ind[0], :]))
        if i_ref:
            info['p_ref_list'].append(_get_one(P, i_ref))

        info['t'] = tpc() - time

        # TODO: move "info" into numpy-cpu before log and return!

        _log(info, log, log_ind, is_new)

    _log(info, log, log_ind, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _check(I, y, info):
    """Check the current batch of function values and save the improvement."""
    ind_opt = jnp.argmax(y) if info['is_max'] else jnp.argmin(y)
    i_opt_curr, y_opt_curr = I[ind_opt, :], y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr
        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(y_opt_curr)
        return True


def _generate_initial(n, r, constr=False):
    """Build initial TT-tensor for probability."""
    d = len(n)

    if constr:
        Y = ind_tens_max_ones(d, 3, r)

    else:
        Y = []
        r = [1] + [r]*(d-1) + [1]
        for i in range(d):
            Y.append(np.random.random(size=(r[i], n[i], r[i+1])))

    return [jnp.array(G) for G in Y]


def _get(Y, I):
    """Compute the elements of the TT-tensor for several multi-indices."""
    Q = Y[0][0, I[:, 0], :]
    for j in range(1, len(Y)):
        Q = jnp.einsum('kq,qkp->kp', Q, Y[j][:, I[:, j], :])
    return Q[:, 0]


def _get_one(Y, i):
    """Compute the element of the TT-tensor for given multi-index."""
    Q = Y[0][0, i[0], :]
    for j in range(1, len(Y)):
        Q = jnp.einsum('q,qp->p', Q, Y[j][:, i[j], :])
    return Q[0]


def _likelihood_one(Y, ind):
    """Likelihood in multi-index ind for TT-tensor Y."""
    d = len(Y)

    Z = [[]] * (d+1)
    Z[-1] = jnp.ones(1)
    for j in range(d-1, 0, -1):
        Z[j] = jnp.sum(Y[j], axis=1) @ Z[j+1]
        Z[j] = Z[j] / jnp.linalg.norm(Z[j])
    Z[0] = Y[0][0, ind[0], :]

    p = jnp.einsum('aib,b->ai', Y[0], Z[1])
    p = p.flatten()
    p = jnp.abs(p)
    p = p / p.sum()

    p_all = [p[ind[0]]]

    for j in range(1, d):
        p = jnp.einsum('a,aib,b->i', Z[j-1], Y[j], Z[j+1])
        p = jnp.abs(p)
        p = p / jnp.sum(p)
        p_all.append(p[ind[j]])

        Z[j] = Z[j-1] @ Y[j][:, ind[j], :]
        Z[j] = Z[j] / jnp.linalg.norm(Z[j])

    return jnp.sum(jnp.log(jnp.array(p_all)))


def _log(info, log=False, log_ind=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = 'protes > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if len(info["y_ref_list"]) > 0:
        text += f' | y_ref {info["y_ref_list"][-1]:-11.4e} | '

    if log_ind:
        text += f' | i {"".join([str(i) for i in info["i_opt"]])}'

    if is_end:
        text += ' <<< DONE'

    print(text)


def _sample_one(Y, key):
    """Generate sample according to given probability TT-tensor."""
    d = len(Y)
    keys = jax.random.split(key, d)
    I = jnp.zeros(d, dtype=jnp.int32)
    Z = [[]] * (d+1)

    Z[-1] = jnp.ones(1)
    for i in range(d-1, 0, -1):
        mat = jnp.sum(Y[i], axis=1)
        Z[i] = mat @ Z[i+1]
        Z[i] = Z[i] / jnp.linalg.norm(Z[i])

    p = jnp.einsum('aib,b->ai', Y[0], Z[1])
    p = p.flatten()
    p = jnp.abs(p)
    p = p / p.sum()

    ind = jax.random.choice(keys[0], jnp.arange(Y[0].shape[1]), p=p)

    Z[0] = Y[0][0, ind, :]
    I = I.at[0].set(ind)

    for i in range(1, d):
        p = jnp.einsum('a,aib,b->i', Z[i-1], Y[i], Z[i+1])
        p = jnp.abs(p)
        p = p / jnp.sum(p)

        ind = jax.random.choice(keys[i], jnp.arange(Y[i].shape[1]), p=p)

        Z[i] = Z[i-1] @ Y[i][:, ind, :]
        Z[i] = Z[i] / jnp.linalg.norm(Z[i])
        I = I.at[i].set(ind)

    return I
