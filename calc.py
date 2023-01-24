import jax.numpy as jnp
import numpy as np
import sys
from time import perf_counter as tpc


from protes import protes
from problems.control import control_build
from problems.control import control_solve_baseline
from problems.func import func_run
from problems.qubo import qubo_build_function
from problems.qubo import qubo_build_matrix
from problems.qubo import qubo_solve_baseline
from utils import Log
from utils import folder_ensure


def calc_control(d=20, M=1.E+3, K=20, k=1, k_gd=50, r=5, lr=1.E-4):
    """Solve the optimal control problem."""
    log = Log(f'result/logs/control.txt')

    txt = f'--> control | '
    txt += f'd={d} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    f, opts = control_build(d)

    t = tpc()
    n_opt = protes(f, d, 2, M, K, k, k_gd, r, lr,
        batch=False, log=True, log_ind=True)
    y_opt = f(n_opt)
    t = tpc() - t

    t_ref = tpc()
    n_opt_ref, y_opt_ref = control_solve_baseline(*opts)
    t_ref = tpc() - t_ref

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e} | Baseline: {y_opt_ref:-14.7e}')
    log(f'Time   : {t:-14.3f} | Baseline: {t_ref:-14.7e}')

    log(f'n opt     >> {"".join([str(n) for n in n_opt])}')
    log(f'n opt ref >> {"".join([str(n) for n in n_opt_ref])}')


def calc_func(d=10, n=50, M=1.E+3, K=20, k=5, k_gd=50, r=5, lr=1.E-4):
    """Perform computations for analytical multivariable functions."""
    log = Log(f'result/logs/func.txt')

    txt = f'--> FUNC | '
    txt += f'd={d} n={n} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    def opt(f):
        return protes(f, d, n, M, K, k, k_gd, r, lr, batch=True, log=False, sig=None)

    func_run(d, n, M, opt, log)


def calc_qubo(d=250, M=1.E+5, K=20, k=1, k_gd=50, r=5, lr=1.E-4):
    """Solve the QUBO problem."""
    log = Log(f'result/logs/qubo.txt')

    txt = f'--> QUBO | '
    txt += f'd={d} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    Q = qubo_build_matrix(d, prob_con=0.5)

    t_ref = tpc()
    y_opt_ref = qubo_solve_baseline(Q)
    t_ref = tpc() - t_ref

    t = tpc()
    f = qubo_build_function(jnp.array(Q))
    n_opt = protes(f, d, 2, M, K, k, k_gd, r, lr,
        batch=True, log=True)
    y_opt = f(n_opt.reshape(1, -1))[0]
    t = tpc() - t

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e} | Baseline: {y_opt_ref:-14.7e}')
    log(f'Time   : {t:-14.3f} | Baseline: {t_ref:-14.7e}')


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'func'

    if mode == 'control':
        calc_control()
    elif mode == 'func':
        calc_func()
    elif mode == 'qubo':
        calc_qubo()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
