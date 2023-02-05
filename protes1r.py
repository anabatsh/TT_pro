import numpy as np
import teneva
import numba
from time import perf_counter as tpc


def protes1r(f, n, M, K=50, k=5, k_gd=500, lr=1.E-3, sig=None, M_ANOVA=None, batch=False,
           with_cache=False, with_qtt=False, is_rand_init=False, log=False, log_ind=False, constr=False,
           ret_tt=False, cores=None, norm=True):
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
        with_qtt (bool): if is True, then QTT-method is used. In this case,
            the tensor mode size "n" should be a power of "2".
        is_rand_init (bool): if is True and "M_ANOVA" is None, then random
            initial approximation will be used. Otherwise, the constant
            TT-tensor will be used.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed every step.
        log_ind (bool): if flag is set and "log" is True, then the current
            optimal multi-index will be printed every step.

    Returns:
        tuple: multi-index "n_opt" (list of the length "d") corresponding to
        the found optimum of the tensor (in the current version only minimum is
        supported) and the related tensor value "y_opt" (float).

    """
    global params, lonly_idx

    d = len(n)

    if constr:
        raise NotImplementedError('Is not supported in colab version!')

    time = tpc()
    # rng = jax.random.PRNGKey(42)
    n_opt = None
    y_opt = np.inf

    info = {'M': 0, 'M_cache': 0}
    cache = {}

    if with_qtt:
        d_base = d
        n_base = n
        q = int(np.log2(n))
        if 2**q != n:
            raise ValueError('Tensor mode size should be power of 2 for QTT')
        d = d * q
        n = 2

    def f_batch(I):
        I = np.array(I)

        f_base_real = f if batch else lambda I: np.array([f(i) for i in I])

        def f_base(I):
            if with_qtt:
                I = teneva.ind_qtt_to_tt(I, q)
            return f_base_real(I)

        if not with_cache:
            info['M'] += I.shape[0]
            return np.array(f_base(I))

        I_new = np.array([np.array(i) for i in I if tuple(i) not in cache])
        if len(I_new):
            Y_new = f_base(I_new)
            for k, i in enumerate(I_new):
                cache[tuple(i)] = Y_new[k]

        info['M'] += len(I_new)
        info['M_cache'] += len(I) - len(I_new)

        return np.array([cache[tuple(i)] for i in I])

    if cores is None:
        params = _generate_initial1r(n, is_rand=is_rand_init)
    else:
        params = cores

    _="""
    optim = optax.adam(lr)
    opt_state = optim.init([jnp.array(i) for i in params])

    
    def make_step_step(params, opt_state, ind):
        #loss_val, grads = jax.value_and_grad(loss)(params, ind, val, y_opt)
        grads = [jnp.array(i) for i in grads1r(params, ind)]
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state
    """
    
    
    prev_idx = [None, 0]
    max_prev_idx = 3
    peaks = []
    history = []
    max_history_size = 15
    #idxs = np.array(peaks)
    idxs_cores = get_constrain_tens([len(pi) for pi in params], peaks)

    def update_peaks(idx, peaks):
        peaks.append(idx)
        idxs_cores = get_constrain_tens([len(pi) for pi in params], peaks)
        #teneva.show(idxs_cores)
        print(teneva.erank(idxs_cores))
        for pi in params:
            pi += 0.01
        norm_p(params)

        return idxs_cores

    
    while True:
        #print(params)
        #rng, key = jax.random.split(rng)
        #key_s = jax.random.split(key, K)
        #arg = [params, idxs] if new_strategy else params
        #ind = generate_random_index(key_s, arg)
        cores = [np.einsum("ijk,j->ijk", G, pi) for G, pi in zip(idxs_cores, params)]
        ind = sample_ind_rand(cores, K, 0.05)
        
        y = f_batch(ind)

        ind_sort = np.argsort(y, kind='stable')
        ind_top = ind[ind_sort[:k], :]

        ind_to_check = ind_top
        ind_np = np.unique(ind_to_check, axis=0)
        
        if False:
            if ind_np.shape[0] == 1:
                lonly_idx = ind_np[0]
                if np.all(lonly_idx == prev_idx[0]):
                    prev_idx[1] += 1
                else:
                    prev_idx = [lonly_idx, 0]

                if prev_idx[1] >= max_prev_idx:
                    print(f'Sampling the same: {lonly_idx} ... ', end='')
                    lonly_idx_t = tuple(lonly_idx)
                    if lonly_idx_t in peaks:
                        print(f'saw this guy before')
                        print(f'VERY BAD!!!')

                    else:
                        idxs_cores = update_peaks(lonly_idx_t, peaks) 
                        #params = _generate_initial1r(d, n, is_rand=is_rand_init)
                        print(f'new blood')

                    continue
                    
        cur_t_top = [tuple(i) for i in ind_top]
        
        if False:
            if len(history) == max_history_size:
                #print(cur_t_top)
                for i_top in cur_t_top:
                    flag = [i_top in i for i in history]
                    #if True in flag:
                    #    print(flag)
                    if all(flag):
                        idxs_cores = update_peaks(i_top, peaks) 

                
        history.append(set(cur_t_top))
        history = history[-max_history_size:]
        
        
        if check_delta(params, th=0.9):
            idxs_cores = update_peaks(get_delta_index(params), peaks)
        

        _="""
        params_jax = [jnp.array(i) for i in params]
        for _ in range(k_gd):
            params_jax, opt_state = make_step_step(params_jax, opt_state,
                ind_top)
         
        params = [np.array(list(i)) for i in params_jax]
        
        """
        make_step(params, ind_top, lr=lr, norm=norm, k_sa=k_gd)

        is_upd = False
        if n_opt is None or np.min(y) < y_opt:
            n_opt = ind[np.argmin(y)]
            y_opt = np.min(y)
            is_upd = True
            
        if False and is_upd and info["M"] > 1e4:
            peaks.append(ind_top[0])
            idxs_cores = get_constrain_tens([len(pi) for pi in params], peaks)
            teneva.show(idxs_cores)
            
            

        if log and (is_upd or info['M'] >= M):
            text = ''
            text += f'Evals : {info["M"]:-7.1e} | '
            # text += f'Cache : {info["M_cache"]:-7.1e} | '
            text += f'Opt : {y_opt:-14.7e} | '
            text += f'Time : {tpc()-time:-7.3f}'
            if log_ind:
                text += f' | n : {"".join([str(n) for n in n_opt])}'
            print(text)

        if info['M'] >= M:
            break

    if with_qtt:
        n_opt = teneva.ind_qtt_to_tt(n_opt, q)

    if ret_tt:
        return n_opt, y_opt, params
    else:
        return n_opt, y_opt




# !!!! TODO change this to the teneva version after update
def sample_ind_rand(Y, m=1, unsert=1e-10):
    """Sample random multi-indices according to given probability TT-tensor.

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        m (int, float): number of samples.
    Returns:
        np.ndarray: generated multi-indices for the tensor in the form
        of array of the shape [m, d], where "d" is the dimension of the tensor.

    """
    d = len(Y)
    res = np.zeros((m, d), dtype=np.int32)
    phi = [None]*(d+1)
    phi[-1] = np.ones(1)
    for i in range(d-1, 0, -1):
        phi[i] = np.sum(Y[i], axis=1) @ phi[i+1]


    p = Y[0] @ phi[1]
    p = p.flatten()
    p += unsert
    p = np.maximum(p, 0)
    p = p/p.sum()
    ind = np.random.choice(Y[0].shape[1], m, p=p)
    phi[0] = Y[0][0, ind, :] # ind here is an array even if m=1
    res[:, 0] = ind
    for i, c in enumerate(Y[1:], start=1):
        p = np.einsum('ma,aib,b->mi', phi[i-1], Y[i], phi[i+1])
        for pi in p:
            pi[np.isnan(pi)] = 0
            pi[np.isinf(pi)] = 0
            if pi.sum() == 0:
                pi[:] = np.ones(len(pi))
            pi /= np.max(np.abs(pi))
            pi[np.isinf(pi)] = 0
            pi[np.isnan(pi)] = 0
            if pi.sum() == 0:
                pi[:] = np.ones(len(pi))

            pi[:] = np.maximum(pi, 0)
        try:
            ind = np.array([np.random.choice(c.shape[1], p=pi/pi.sum()) for pi in p])
        except:
            print(p)
            exit(0)
        res[:, i] = ind
        phi[i] = np.einsum("il,lij->ij", phi[i-1], c[:, ind])

    return res


def get_constrain_tens(n, idxs):
    res = [np.ones([1, ni, 1]) for ni in n]
    for idx in idxs:
        if len(idx) > 0:
            cur_t = teneva.tensor_delta(n, idx, -1)
            res = teneva.add(res, cur_t)

    # norm it
    for G in res:
        G /= G.shape[1]

    return res

def build_z(p, idxs):
    n = [len(pi) for pi in p]
    #d = len(p)
    #n = len(p[0])
    t = get_constrain_tens(n, idxs)
    return [np.einsum("ijk,j->ijk", ti, pi) for ti, pi in zip(t, p)]


def norm_p(p):
    for pi in p:
        pi /= pi.sum()

@numba.jit
def sa_log(p, lr, k):
    for _ in range(k):
        p = p + lr/p
        
    return p
        
def make_step(p, top_idx, lr=1e-4, norm=False, k_sa=10):
    top_idx = np.asarray(top_idx)
    for pi, good_idxs in zip(p, top_idx.T):
        unique, counts = np.unique(good_idxs, return_counts=True)
        for u, step in zip(unique, lr*counts):
            pi[u] = sa_log(pi[u], step, k_sa)
            #pi[u] = sa_log(pi[u], lr, k_sa)
            
        
    if norm:
        norm_p(p)


    
def grads1r(p, top_idx):
    res = [jnp.zeros(len(pi)) for pi in p]

    top_idx = np.asarray(top_idx)
    for pi, good_idxs, ri in zip(p, top_idx.T, res):
        unique, counts = jnp.unique(good_idxs, return_counts=True)
        for u, cnt in zip(unique, counts):
            ri.at[u].set(cnt/pi[u])
    
    return res

        
def _generate_initial1r(n, *, is_rand=True, noise_not_rand=1e-2):
    """Build initial TT-tensor for probability."""
    if is_rand:
        Y = [np.random.random(size=ni) for ni in n]
    else:
        Y = [(1 + noise_not_rand*np.random.random(size=ni)) for ni in n] 

    for pi in Y:
        pi /= pi.sum()

    return Y


def check_delta(p, th=0.9):
    return all([np.max(pi) > th for pi in p])

def get_delta_index(p):
    return np.array([np.argmax(pi) for pi in p])





