import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import teneva
from time import perf_counter as tpc


def protes(f, d, n, M, K=20, k=1, k_gd=50, r=5, lr=1.E-4, sig=1.E-1, M_ANOVA=None, batch=False, with_cache=False, is_rand_init=False, log=False, log_ind=False):
    """Tensor optimization based on sampling from the probability TT-tensor.

    Method PROTES (PRobability Optimizer with TEnsor Sampling) for optimization
    of the multidimensional arrays and  discretized multivariable functions
    based on the tensor train (TT) format.

    Args:
        f (function): the target function "f(I)", where input "I" is a 2D
            np.ndarray of the shape "[samples, d]" ("d" is a number of
            dimensions of the function's input). The function should return 1D
            np.ndarray of the length equals to "samples" (the values of the
            target function for all provided multi-indices). If "batch" flag is
            False, then function looks like "f(i)", where input "i" is an 1D
            np.ndarray of the shape "[d]" and the output is a number.
        d (int): number of the tensor dimensions.
        n (int): tensor size for each dimension (it is equal for each mode).
        M (int): the number of allowed requests to the objective function (> 0).
        K (int): the batch size for optimization.
        k (int): number of selected candidates for all batches (< K).
        k_gd (int): number of GD iterations for each batch.
        r (int): TT-rank of the constructed probability TT-tensor.
        lr (float): learning rate for GD.
        sig (float): parameter for exponential in loss function. If is None,
            then base method with "top-k" candidates will be used.
        M_ANOVA (int): number of requests used for TT-ANOVA initial
            approximation. If it is zero or None, then constant initial TT
            tensor will be used.
        batch (bool): if is True, then function "f" has 2D dimensional input
            (several samples). Otherwise, the input is one-dimensional.
        with_cache (bool): if is True, then cache for requested function values
            will be used.
        is_rand_init (bool): if is True and "M_ANOVA" is None, then random
            initial approximation will be used. Otherwise, the constant
            TT-tensor will be used.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed every step.
        log_ind (bool): if flag is set and "log" is True, then the current
            optimal multi-index will be printed every step.

    Returns:
        list: multi-index corresponding to the found optimum of the tensor (in
        the current version only minimum is supported).

    """
    time = tpc()
    rng = jax.random.PRNGKey(42)
    n_opt = None
    y_opt = jnp.inf

    info = {'M': 0, 'M_cache': 0}
    cache = {}

    def f_batch(I):
        I = np.array(I)
        f_base = f if batch else lambda I: np.array([f(i) for i in I])

        if not with_cache:
            info['M'] += I.shape[0]
            return jnp.array(f_base(I))

        I_new = np.array([np.array(i) for i in I if tuple(i) not in cache])
        if len(I_new):
            Y_new = f_base(I_new)
            for k, i in enumerate(I_new):
                cache[tuple(i)] = Y_new[k]

        info['M'] += len(I_new)
        info['M_cache'] += len(I) - len(I_new)

        return jnp.array([cache[tuple(i)] for i in I])


    params = _generate_initial(d, n, r, f_batch, M_ANOVA, is_rand_init)
    generate_random_index = _build_generate_random_index()
    optim = optax.adam(lr)
    opt_state = optim.init(params)
    make_step = _build_make_step(optim, sig)

    while True:
        rng, key = jax.random.split(rng)
        key_s = jax.random.split(key, K)
        ind = generate_random_index(key_s, params)
        y = f_batch(ind)

        ind_sort = np.argsort(y, kind='stable')
        ind_top = ind[ind_sort[:k], :]

        for _ in range(k_gd):
            loss_val, params, opt_state = make_step(params, opt_state,
                ind, y, y_opt if y_opt < 1.E+10 else 0.)

        is_upd = False
        if n_opt is None or jnp.min(y) < y_opt:
            n_opt = ind[jnp.argmin(y)]
            y_opt = jnp.min(y)
            is_upd = True

        if log and (is_upd or info['M'] >= M):
            text = ''
            text += f'Evals : {info["M"]:-7.1e} | '
            text += f'Cache : {info["M_cache"]:-7.1e} | '
            text += f'Opt : {y_opt:-14.7e} | '
            text += f'Time : {tpc()-time:-7.3f}'
            if log_ind:
                text += f' | n : {"".join([str(n) for n in n_opt])}'
            print(text)

        if info['M'] >= M:
            break

    return n_opt


def _build_generate_random_index():
    """Sample random multi-index from probability TT-tensor."""
    def generate_random_index(key, z):
        d = len(z)
        keys = jax.random.split(key, d)
        res = jnp.zeros(d, dtype=jnp.int32)
        phi = [[]] * (d+1)
        phi[-1] = jnp.ones(1)
        for i in range(d-1, 0, -1):
            mat = jnp.sum(z[i], axis=1)
            phi[i] = mat @ phi[i+1]
            phi[i] = phi[i] / jnp.linalg.norm(phi[i])

        p = jnp.einsum('aib,b->ai', z[0], phi[1])
        p = p.flatten()
        p = jnp.abs(p)
        p = p / p.sum()

        ind = jax.random.choice(keys[0], jnp.arange(z[0].shape[1]), p=p)

        phi[0] = z[0][0, ind, :]
        res = res.at[0].set(ind)

        for i in range(1, d):
            p = jnp.einsum('a,aib,b->i', phi[i-1], z[i], phi[i+1])
            p = jnp.abs(p)
            p = p / jnp.sum(p)
            ind = jax.random.choice(keys[i], jnp.arange(z[i].shape[1]), p=p)
            mat = z[i][:, ind, :]
            phi[i] = phi[i-1] @ mat
            phi[i] = phi[i] / jnp.linalg.norm(phi[i])
            res = res.at[i].set(ind)

        return res

    return jax.jit(jax.vmap(generate_random_index, (0, None)))


def _build_make_step(optim, sig=None):
    """Perform GD step for probability TT-tensor."""
    compute_likelihood = jax.jit(jax.vmap(_likelihood, (0, None)))

    @jax.jit
    def loss_old(z, ind_top, val=None, y_opt=None):
        l = -compute_likelihood(ind_top, z)
        return jnp.mean(l)

    @jax.jit
    def loss_new(z, ind, val, y_opt):
        f = jnp.exp(-1./sig * (val-jnp.min(val)))
        #f = jnp.exp(-1./sig * (val-y_opt))
        p = compute_likelihood(ind, z)
        l = f * p
        return -jnp.mean(l)

    loss = loss_new if sig else loss_old

    @jax.jit
    def make_step(params, opt_state, ind_top, val, y_opt):
        loss_val, grads = jax.value_and_grad(loss)(params, ind_top, val, y_opt)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss_val, params, opt_state

    return make_step


def _generate_initial(d, n, r, f=None, M=None, is_rand=True):
    """Build initial TT-tensor for probability."""
    if f is None or M is None or M < 1:
        if is_rand:
            # Initial approximation with random TT-tensor:
            rs = [1] + [r]*(d-1) + [1]
            Y = []
            for i in range(d):
                if is_rand:
                    Y.append(np.random.random(size=(rs[i], n, rs[i+1])))
        else:
            # Initial approximation with constant TT-tensor:
            Y = teneva.tensor_const([n]*d, 1.)
            for _ in range(r-1):
                Y = teneva.add(Y, teneva.tensor_const([n]*d, 1.))
            Y = teneva.mul(Y, 1./r)

    else:
        # Initial approximation with TT-ANOVA:
        I_trn = teneva.sample_lhs([n]*d, int(M))
        Y_trn = f(I_trn)
        Y_trn = 1./(1.E-4 + Y_trn)
        Y_trn = Y_trn**4
        Y = teneva.ANOVA(I_trn, Y_trn).cores(r=r, rel_noise=0.1)

    return [jnp.array(G) for G in Y]


def _likelihood(ind, z):
    """Likelihood in multi-index ind for TT-tensor z."""
    d = len(z)
    res = jnp.zeros(d, dtype=jnp.int32)
    phi = [[]] * (d+1)
    phi[-1] = jnp.ones(1)

    for i in range(d-1, 0, -1):
        mat = jnp.sum(z[i], axis=1)
        phi[i] = mat @ phi[i+1]
        phi[i] = phi[i] / jnp.linalg.norm(phi[i])
    phi[0] = z[0][0, ind[0], :]

    p = jnp.einsum('aib,b->ai', z[0], phi[1])
    p = p.flatten()
    p = jnp.abs(p)
    p = p / p.sum()

    p_all = [p[ind[0]]]

    for i in range(1, d):
        p = jnp.einsum('a,aib,b->i', phi[i-1], z[i], phi[i+1])
        p = jnp.abs(p)
        p = p / jnp.sum(p)
        p_all.append(p[ind[i]])

        mat = z[i][:, ind[i], :]
        phi[i] = phi[i-1] @ mat
        phi[i] = phi[i] / jnp.linalg.norm(phi[i])

    return jnp.sum(jnp.log(jnp.array(p_all)))
