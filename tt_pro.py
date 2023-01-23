import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import teneva
from time import perf_counter as tpc


def build_generate_random_index():
    def generate_random_index_(key, z):
        # z - это тензор вероятности в TT-формате,
        # функция возвращает один вероятностно сгенерированный мульти-индекс

        d = len(z)
        keys = jax.random.split(key, d)
        res = jnp.zeros(d, dtype=jnp.int32)
        phi = [[]]*(d+1)
        phi[-1] = jnp.ones(1)
        for i in range(d-1, 0, -1):
            mat = jnp.sum(z[i], axis=1)
            phi[i] = mat@phi[i+1]
            phi[i] = phi[i]/jnp.linalg.norm(phi[i])

        p = jnp.einsum('aib,b->ai', z[0], phi[1])
        p = p.flatten()
        p = jnp.abs(p)
        p = p/p.sum()

        ind = jax.random.choice(keys[0], jnp.arange(z[0].shape[1]), p=p)

        phi[0] = z[0][0, ind, :]
        res = res.at[0].set(ind)

        for i in range(1, d):
            p = jnp.einsum('a,aib,b->i', phi[i-1], z[i], phi[i+1])
            p = jnp.abs(p)
            p = p/jnp.sum(p)
            ind = jax.random.choice(keys[i], jnp.arange(z[i].shape[1]), p=p)
            mat = z[i][:, ind, :]
            phi[i] = phi[i-1]@mat
            phi[i] = phi[i]/jnp.linalg.norm(phi[i])
            res = res.at[i].set(ind)

        return res

    return jax.jit(jax.vmap(generate_random_index_, (0, None)))


def build_make_step(optim):
    compute_likelihood = jax.jit(jax.vmap(likelihood, (0, None)))

    def loss(z, ind1):
        l = -compute_likelihood(ind1, z)
        # l += compute_likelihood(ind2, z)
        return jnp.mean(l)

    @jax.jit
    def make_step(params, ind1, opt_state):
        loss_val, grads = jax.value_and_grad(loss)(params, ind1)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss_val, params, opt_state

    return make_step


def generate_initial(d, n, r, f=None, M=None):
    if f is None or M is None or M < 1:
        # Initial approximation with random TT-tensor:
        rs = [1] + [r]*(d-1) + [1]
        Y = []
        for i in range(d):
            Y.append(np.random.random(size=(rs[i], n, rs[i+1])))

    else:
        # Initial approximation with TT-ANOVA:
        I_trn = teneva.sample_lhs([n]*d, int(M))
        Y_trn = f(I_trn)
        Y_trn = 1./(1.E-4 + Y_trn)
        Y_trn = Y_trn**4
        Y = teneva.ANOVA(I_trn, Y_trn).cores(r=r, rel_noise=0.1)

    return [jnp.array(G) for G in Y]


def likelihood(ind, z):
    # Функция для вычисления правдоподобия в мульти-индексе ind для TT-тензора z:

    d = len(z)
    res = jnp.zeros(d, dtype=jnp.int32)
    phi = [[]]*(d+1)
    phi[-1] = jnp.ones(1)

    for i in range(d-1, 0, -1):
        mat = jnp.sum(z[i], axis=1)
        phi[i] = mat@phi[i+1]
        #nrm = jnp.linalg.norm(phi)
        phi[i] = phi[i]/jnp.linalg.norm(phi[i])
        #mat z[i] = z[i]/nrm

    p = jnp.einsum('aib,b->ai', z[0], phi[1])
    p = p.flatten()
    p = jnp.abs(p)
    p = p/p.sum()
    phi[0] = z[0][0, ind[0], :]
    p_all = [p[ind[0]]]

    for i in range(1, d):
        p = jnp.einsum('a,aib,b->i', phi[i-1], z[i], phi[i+1])
        p = jnp.abs(p)
        p = p/jnp.sum(p)
        mat = z[i][:, ind[i], :]
        phi[i] = phi[i-1]@mat
        phi[i] = phi[i]/jnp.linalg.norm(phi[i])
        p_all.append(p[ind[i]])

    return jnp.sum(jnp.log(jnp.array(p_all)))


def tt_pro(f, d, n, M, K, k, k_gd, r, M_ANOVA=None, info={}, batch=False, log=False):
    time = tpc()
    rng = jax.random.PRNGKey(42)
    f_batch = f if batch else lambda I: np.array([f(i) for i in I])
    M_cur = int(M_ANOVA or 0)
    n_opt = None
    y_opt = jnp.inf

    params = generate_initial(d, n, r, f_batch, M_ANOVA)
    generate_random_index = build_generate_random_index()
    optim = optax.adam(1.E-4)
    opt_state = optim.init(params)
    make_step = build_make_step(optim)

    while(True):
        rng, key = jax.random.split(rng)
        key_s = jax.random.split(key, K)
        ind = generate_random_index(key_s, params)
        y = f_batch(ind)
        M_cur += K

        ind_sort = np.argsort(y, kind='stable')
        ind_top = ind[ind_sort[:k], :]

        for _ in range(k_gd):
            loss_val, params, opt_state = make_step(params, ind_top, opt_state)

        is_upd = False
        if n_opt is None or jnp.min(y) < y_opt:
            n_opt = ind[jnp.argmin(y)]
            y_opt = jnp.min(y)
            is_upd = True

        if log and (is_upd or M_cur >= M):
            text = ''
            text += f'Evals : {M_cur:-7.1e} | '
            text += f'Opt : {y_opt:-14.7e} | '
            text += f'Time : {tpc()-time:-14.3f}'
            print(text)

        if M_cur >= M:
            break

    info['t'] = tpc()-time

    return n_opt
