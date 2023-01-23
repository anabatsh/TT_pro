from copy import deepcopy
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import teneva
from time import perf_counter as tpc


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


def compute_likelihood_(ind, z):
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


def build_initial(d, n, r):
    rs = [1]+[r]*(d-1)+[1]
    q0 = []
    for i in range(d):
        q0.append(np.random.random(size=(rs[i], n, rs[i+1])))
    return [jnp.array(q1) for q1 in q0]

def build_initial_ANOVA(shape, r, f, M):
    I = teneva.sample_lhs(shape, M)
    Y = [f(x)**4 for x in I]
    q0 = teneva.ANOVA(I, Y).cores(r=r, rel_noise=0.1)

    return [jnp.array(q1) for q1 in q0]




def tt_pro(f, d, n, M, K, k, k_sgd, r=5, batch=False, log=False, f_ANOVA=None, M_ANOVA=None):
    rng = jax.random.PRNGKey(42)
    y_opt = jnp.inf

    if M_ANOVA is None:
        params = build_initial(d, n, r)
    else:
        params = build_initial_ANOVA([n]*d, r, f_ANOVA, M_ANOVA)
        M -= M_ANOVA

    optim = optax.adam(1.E-4)
    opt_state = optim.init(params)

    generate_random_index = jax.jit(jax.vmap(generate_random_index_, (0, None)))
    compute_likelihood = jax.jit(jax.vmap(compute_likelihood_, (0, None)))

    def loss(z, ind1):
        lk = -compute_likelihood(ind1, z) # + compute_likelihood(ind2, z)
        return jnp.mean(lk)

    @jax.jit
    def make_step(params, ind1, opt_state):
        loss_val, grads = jax.value_and_grad(loss)(params, ind1)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss_val, params, opt_state

    n_opt = None

    for m in range(int(M/K)):
        rng, key = jax.random.split(rng)
        key_s = jax.random.split(key, K)
        ind = generate_random_index(key_s, params)

        if batch:
            y = f(ind)
        else:
            y = np.array([f(i) for i in ind])

        ind_sort = np.argsort(y)
        ind_top = ind[ind_sort[:k], :]

        for _ in range(k_sgd):
            loss_val, params, opt_state = make_step(params, ind_top, opt_state)

        is_upd = False
        if n_opt is None or jnp.min(y) < y_opt:
            n_opt = ind[jnp.argmin(y)]
            y_opt = jnp.min(y)
            is_upd = True

        if log and is_upd:
            print(f'Evals : {m*K:-7.1e} | Opt : {y_opt:-14.7e}')

    return n_opt
