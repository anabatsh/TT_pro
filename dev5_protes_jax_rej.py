import jax
import jax.numpy as np
import optax
from time import perf_counter as tpc
import numpy as onp
import teneva

import os
from datetime import datetime

def save_raw_data(**data):
    now = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")

    np.savez(f"raw_data_{os.getpid()}_{now}", **data)

def cached_func(f, info):
    cache = dict()
    def fn(I):
        if I.ndim > 1:
            I_new = [i for i in I.tolist() if tuple(i) not in cache]

            if len(I_new) > 0:
                Y_new = f(np.array(I_new))
                for i, y in zip(I_new, Y_new):
                    cache[tuple(i)] = y

            info['m'] += len(I_new)
            info['M_cache'] += len(I) - len(I_new)

            y = np.array([cache[tuple(i)] for i in I.tolist()])
            return y
        else: # just 1 point
            print("Not implemented")
    fn.cache = cache
    return fn


def apply_const(Y, cnstr):
    if cnstr is not None:
        Y = mul(Y, cnstr)

    Y = _orthogonalize(Y, use_stab=False, orht_fst=True)
    return Y


def sample_from_batch_iter(P, rng, sample, k=100):
    pul_I = []
    pul_maxp = []
    while True:
        if len(pul_I) == 0:
            rng, key = jax.random.split(rng)
            pul_I, pul_maxp = sample(P, jax.random.split(key, k))

        yield pul_I[0:1], pul_maxp[0:1]
        pul_I = pul_I[1:]
        pul_maxp[1:]


def protes_jax_rej(f, n, m, k_gd=100, lr=1.E-4, r=2, T=1., how_to_upd=True, P=None, seed=42, info={}, i_ref=None, is_max=False, log=False, log_ind=False, mod='jax', device='cpu', K_rebuild=300):
    time = tpc()
    info.update({'mod': mod, 'is_max': is_max, 'm': 0, 't': 0, 'M_cache': 0,
        'i_opt': None, 'y_opt': None, 'm_opt_list': [], 'y_opt_list': [],
        'm_ref_list': [],
        'p_ref_list': [], 'p_opt_ref_list': [], 'p_top_ref_list': []})

    rng = jax.random.PRNGKey(seed)

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(n, r, key)
        rng, keyP = jax.random.split(rng)

    sample = jax.jit(jax.vmap(_sample, (None, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood, (None, 0)))

    @jax.jit
    def loss(P_cur, I_cur):
        return np.mean(-likelihood(P_cur, I_cur))

    loss_grad = jax.grad(loss)

    @jax.jit
    def optimize(P, I_cur):
        grads = loss_grad(P, I_cur)
        res = [update_orth_Wood(P[0].reshape(-1, 1), grads[0].reshape(-1, 1), lr=lr).reshape(*P[0].shape) ]
        for X, G in zip(P[1:], grads[1:]):
            r1, n1, r2 = X.shape
            core = update_orth_Wood(X.reshape(r1, n1*r2).T, G.reshape(r1, n1*r2).T, lr=lr).T
            res.append(core.reshape(r1, n1, r2))

        #jax.debug.print("🤯 {P} {G} {R} 🤯", P=P[0], G=grads[0], R=res[0])


        return res

    peaks = []
    shapes = [pi.shape[1] for pi in P] 

    # idxs_cores = get_constrain_tens(shapes, peaks)
    # all_cores = P
    rng, key = jax.random.split(rng)
    sample_from_batch = sample_from_batch_iter(P, key, sample)

    f = cached_func(f, info)

    # TODO hardcore it!!!
    k = 1

    prev = None
    was_accept = True
    history_sample = []
    history_sample_length = 20

    while True:
        flag = True
        cnt = 0
        while flag:
            # rng, key = jax.random.split(rng)
            # I, max_p = sample(all_cores, jax.random.split(key, k))
            I, max_p = next(sample_from_batch)

            I0_list = I[0].tolist()

            cnt += 1
            flag = (I0_list in peaks) and (cnt < 20)
        
        history_sample.append(I0_list)
        history_sample = history_sample[-history_sample_length:]


        # Iu = np.unique(I, axis=0)
        # Iu = I
        # if np.min(max_p) > 0.95: # thr p is an empirical value
        if cnt == 20 or (len(history_sample) == history_sample_length and onp.unique(history_sample, axis=0).shape[0] == 1):
            pI = I0_list
            if pI in peaks:
                # save_raw_data(I0=I0_list, P=P, idxs_cores=idxs_cores, reason="p")
                print(f"Again in the same local minimum: {pI}")
            else:
                peaks.append(pI)

            # idxs_cores = get_constrain_tens(shapes, peaks)
            I_big_trn = most_k_cache(f.cache, [], k=K_rebuild + len(peaks))
            # P = _generate_initial1r(n, is_rand=is_rand_init, sq=sq)
            keyP, key = jax.random.split(keyP)
            P =  _generate_initial(n, r, key)
            for _ in range(k_gd):
                P = optimize(P, I_big_trn)

            all_cores = P
            rng, key = jax.random.split(rng)
            sample_from_batch = sample_from_batch_iter(P, key, sample)

            val_p =  f(np.array([ peaks[-1] ]))
            print(f"Всё, заело, m {info['m']} | cache {info['M_cache']} |  number of peak: {len(peaks)} | max_p : {np.min(max_p)} ,  idx: \n [{''.join([ str(i) for i in peaks[-1]])}], val: {val_p}")
            # print(f"cur peaks: {peaks}")
            continue
            #exit(0)


        #####
        y = f(I)
        #######

        is_new = _check(I, y, info)

        if info['m'] >= m:
            break

        # ind = np.argsort(y, kind='stable')
        # ind = (ind[::-1] if is_max else ind)[:k_top]

        I0 = I[0]
        y0 = y[0]

        log_like_0 = likelihood(P, I)[0]

        ## rejection!
        if prev is not None:
            # print(prev)
            I_prev, y_prev, log_like_prev = prev
            f_prev = f.cache[tuple(I_prev.tolist())]
            f0 = f.cache[tuple(I0.tolist())]

            pi_x_new_div_x_log = -(f0 - f_prev)/T
            pi_star_x_div_new_x_log = log_like_prev - log_like_0
            # print(pi_star_x_div_new_x_log)
            pi_star_x_div_new_x_log = 0

            alpha = min(np.exp(pi_x_new_div_x_log + pi_star_x_div_new_x_log), 1.)

            rng, key = jax.random.split(rng)
            was_accept = jax.random.uniform(key) < alpha
            if was_accept: # accept
                prev = (I0, y0, log_like_0)
                # print("A")
            else:
                I0, y0, log_like_0 = prev
                # print("R")

        else:
            prev = (I0, y0, log_like_0)


        # if was_accept:
        if how_to_upd or was_accept:
            I = np.array([I0])
            for _ in range(k_gd):
                P = optimize(P, I)

            all_cores = P
            rng, key = jax.random.split(rng)
            sample_from_batch = sample_from_batch_iter(P, key, sample)


        if i_ref is not None: # For debug only
            _set_ref(P, info, I, ind, i_ref)

        info['t'] = tpc() - time

        _log(info, log, log_ind, is_new)

    _log(info, log, log_ind, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _check(I, y, info):
    """Check the current batch of function values and save the improvement."""
    ind_opt = np.argmax(y) if info['is_max'] else np.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr
        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(y_opt_curr)
        return True


def _generate_initial(n, r, key):
    """Build initial random TT-tensor for probability."""
    d = len(n)
    r = [1] + [r]*(d-1) + [1]
    keys = jax.random.split(key, d)

    Y = []
    for j in range(d):
        Y.append(jax.random.uniform(keys[j], (r[j], n[j], r[j+1])))

    return _orthogonalize(Y, use_stab=False, orht_fst=True)
    # return _orthogonalize(Y, use_stab=True, orht_fst=True)


def _get(Y, i):
    """Compute the element of the TT-tensor Y for given multi-index i."""
    Q = Y[0][0, i[0], :]
    for j in range(1, len(Y)):
        Q = np.einsum('r,rq->q', Q, Y[j][:, i[j], :])
    return Q[0]


def _get_many(Y, K):
    """Compute the elements of the TT-tensor on many indices.

    Args:
        Y (list): d-dimensional TT-tensor.
        K (list of list, np.ndarray): the multi-indices for the tensor in the
            form of a list of lists or array of the shape [samples, d].

    Returns:
        np.ndarray: the elements of the TT-tensor for multi-indices "K" (array
        of length "samples").

    """
    Q = Y[0][0, K[:, 0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('kq,qkp->kp', Q, Y[i][:, K[:, i], :])
    return Q[:, 0]

def _interface_matrices(Y):
    """Compute the "interface matrices" for the TT-tensor Y."""
    d = len(Y)
    Z = [[]] * (d+1)
    Z[0] = np.ones(1)
    Z[d] = np.ones(1)
    for j in range(d-1, 0, -1):
        Z[j] = np.sum(Y[j], axis=1) @ Z[j+1]
        Z[j] /= np.linalg.norm(Z[j])
    return Z


def _likelihood_old(Y, I):
    """Compute the likelihood in a multi-index I for TT-tensor Y."""
    d = len(Y)

    Z = _interface_matrices(Y)

    G = np.einsum('riq,q->i', Y[0], Z[1])
    G = np.abs(G)
    G /= G.sum()

    y = [G[I[0]]]

    Z[0] = Y[0][0, I[0], :]

    for j in range(1, d):
        G = np.einsum('r,riq,q->i', Z[j-1], Y[j], Z[j+1])
        G = np.abs(G)
        G /= np.sum(G)

        y.append(G[I[j]])

        Z[j] = Z[j-1] @ Y[j][:, I[j], :]
        Z[j] /= np.linalg.norm(Z[j])

    return np.sum(np.log(np.array(y)))

def _likelihood(Y, I):
    d = len(Y)

    G = Y[0][0, :, :]
    G = np.sum(G**2, axis=1)
    # G /= G.sum() ##???? to remove?

    y = [G[I[0]]]


    Z = Y[0][0, I[0], :]

    norms = []

    for j in range(1, d):
        G = np.einsum('r,riq->iq', Z, Y[j])
        G = np.sum(G**2, axis=1)
        # G /= np.sum(G) ##???? to remove?

        y.append(G[I[j]])

        Z = Z @ Y[j][:, I[j], :]
        Zn = np.linalg.norm(Z)
        norms.append(Zn)
        Z /= Zn

    # jax.debug.print("🤯 Y: {Y} norms: {n} 🤯", Y=Y[:3], n=norms)

    return np.sum(np.log(np.array(y))) + np.sum(np.log(np.array(norms[:-1])))

def _log(info, log=False, log_ind=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes-{info["mod"]} > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if len(info["p_ref_list"]) > 0:
        text += f' | p_ref {info["p_ref_list"][-1]:-11.4e} | '

    if log_ind:
        text += f' | i {"".join([str(i) for i in info["i_opt"]])}'

    if is_end:
        text += ' <<< DONE'

    print(text)


def _sample_abs(Y, key):
    """Generate sample according to given probability TT-tensor Y."""
    d = len(Y)
    keys = jax.random.split(key, d)
    I = np.zeros(d, dtype=np.int32)

    Z = _interface_matrices(Y)

    G = np.einsum('riq,q->i', Y[0], Z[1])
    G = np.abs(G)
    G /= G.sum()

    i = jax.random.choice(keys[0], np.arange(Y[0].shape[1]), p=G)
    I = I.at[0].set(i)

    Z[0] = Y[0][0, i, :]

    for j in range(1, d):
        G = np.einsum('r,riq,q->i', Z[j-1], Y[j], Z[j+1])
        G = np.abs(G)
        G /= np.sum(G)

        i = jax.random.choice(keys[j], np.arange(Y[j].shape[1]), p=G)
        I = I.at[j].set(i)

        Z[j] = Z[j-1] @ Y[j][:, i, :]
        Z[j] /= np.linalg.norm(Z[j])

    return I

# def _sample(Y, key, cnstr):
def _sample(Y, key):
    """Generate sample according to given probability TT-tensor Y."""
    d = len(Y)

    # if cnstr is not None:
        # Y = mul(Y, cnstr)
        # Y = _orthogonalize(Y, use_stab=False, orht_fst=True)


    keys = jax.random.split(key, d)
    I = np.zeros(d, dtype=np.int32)

    G = np.sum(Y[0][0]**2, axis=1)
    G /= G.sum()


    # is_delta = np.zeros(d, dtype=np.int32)
    is_delta = np.zeros(d)

    i = jax.random.choice(keys[0], np.arange(Y[0].shape[1]), p=G)
    is_delta = is_delta.at[0].set(np.max(G))


    I = I.at[0].set(i)

    Z = Y[0][0, i, :]

    for j in range(1, d):
        G = np.einsum('r,riq->iq', Z, Y[j])
        G = np.sum(G**2, axis=1)
        G /= np.sum(G)

        i = jax.random.choice(keys[j], np.arange(Y[j].shape[1]), p=G)
        is_delta = is_delta.at[j].set(np.max(G))
        I = I.at[j].set(i)

        Z = Z @ Y[j][:, i, :]
        Z /= np.linalg.norm(Z)
        # jax.debug.print("🤯 {j}: {p} 🤯", j=j, p=Z)


    # jax.debug.print("🤯 {p} 🤯", p=is_delta)
    # if is_delta.sum() == d:
        # jax.debug.print("🤯 Converged to delta, index: {p} 🤯", p=I)

    return I, is_delta


def _set_ref(P, info, I, ind, i_ref=None):
    info['m_ref_list'].append(info['m'])
    info['p_opt_ref_list'].append(_get(P, info['i_opt']))
    info['p_top_ref_list'].append(_get(P, I[ind[0], :]))
    if i_ref is not None:
        info['p_ref_list'].append(_get(P, i_ref))



def _orthogonalize(Z, use_stab=False, orht_fst=True):
    for i in range(len(Z)-1, 0, -1):
        r2, n2, r3 = Z[i].shape
        G2 = np.reshape(Z[i], (r2, n2 * r3), order='F')
        # R, Q = jsp.linalg.rq(G2.T, mode='reduced')
        # jax.debug.print("🤯 {p} 🤯", p=G2.shape)
        Q, R = np.linalg.qr(G2.T, mode='reduced')
        R = R.T
        Q = Q.T
        Z[i] = np.reshape(Q, (Q.shape[0], n2, r3), order='F')

        r1, n1, r2 = Z[i-1].shape
        G1 = np.reshape(Z[i-1], (r1 * n1, r2), order='F')
        G1 = G1 @ (R / np.linalg.norm(R))
        Z[i-1] = np.reshape(G1, (r1, n1, G1.shape[1]), order='F')
        if use_stab:
            Z[i-1], _ = _core_stab(Z[i-1])


    # print(Z[0])
    if orht_fst:
        Z[0] /= np.linalg.norm(Z[0])

    return Z


def update_orth(X, G, lr=1e-3):
    A = G @ X.T - X @ G.T
    I = np.eye(A.shape[0])
    Q = np.linalg.inv(I + lr/2*A) @ (I - lr/2*A)
    #Q = np.linalg.solve(I + lr/2*A,  I - lr/2*A)
    return Q @ X


def update_orth_Wood(X, G, lr=1e-3):
    U = np.hstack([G, -X])
    V = np.vstack([X.T, G.T])

    I = np.eye(U.shape[0])
    #B_inv = I - U @  np.linalg.inv(np.eye(V.shape[0]) + V@U*lr/2) @ V*lr/2
    B_inv = I - lr/2*U @  np.linalg.solve(np.eye(V.shape[0]) + V@U*lr/2,  V)
    Q = B_inv  @ (I - lr/2*U@V)
    return Q @ X


def _core_stab(G, p0=0, thr=1.E-100):
    """Scaling for the passed TT-core, i.e., G -> (Q, p), G = 2^p * Q.

    Args:
        G (np.ndarray): TT-core in the form of 3-dimensional array.
        p0 (int): optional initial value of the power-factor (it will be added
            to returned value "p").
        thr (float): threshold value for applying scaling (if the maximum
            modulo element in the TT-core is less than this value, then scaling
            will not be performed).

    Returns:
        (np.ndarray, int): scaled TT-core (Q) and power-factor (p), such that
            G = 2^p * Q.

    """
    v_max = np.max(np.abs(G))

    # if v_max <= thr:
        # return G, p0

    p = (np.floor(np.log2(v_max))).astype(int)
    Q = G / 2.**p

    return Q, p0 + p



def get_constrain_tens(n, idxs):
    res = [onp.ones([1, ni, 1]) for ni in n]
    for idx in idxs:
        idx = onp.array(list(idx))
        if len(idx) > 0:
            cur_t = teneva.delta(n, idx, -1)
            res = teneva.add(res, cur_t)

    # teneva.show(res)
    return [np.array(i) for i in res]


def mul_new(Y1, Y2):
    return [G1[:, None, :, :, None] * G2[None, :, :, None, :].reshape(
           [G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
               for G1, G2 in zip(Y1, Y2)]




def mul(Y1, Y2):
    Y = []
    for G1, G2 in zip(Y1, Y2):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        Y.append(G)

    return Y

def most_k_cache(cache, bad, k=100):
    for i in cache:
        j = i
        break


    K = len(cache)
    all_I = np.empty([K, len(j)], dtype=np.int32)
    y = np.empty(K)

    bad_set = set([tuple(i) for i in bad])

    cnt = 0
    for X, Y in cache.items():
        if X in bad_set:
            continue
        all_I = all_I.at[cnt].set(X)
        y = y.at[cnt].set(Y)
        cnt += 1

    idx = np.argsort(y[:cnt])
    return all_I[idx[:k]]


